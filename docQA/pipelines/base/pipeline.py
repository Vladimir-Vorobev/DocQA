from .base_pipeline import BasePipeline
from docQA.nodes.storage import Storage
from docQA.typing_schemas import PipeOutput
from docQA.metrics import top_n_qa_error
from docQA.errors import PipelineError

from typing import Union, List, Any
import json


class Pipeline(BasePipeline):
    def __init__(
            self,
            storage: Storage = None
    ):
        super().__init__()
        self.storage = storage
        self.nodes = {}

    def __call__(
            self,
            data: Any,
            return_translated: bool = False,
            threshold: float = 0.3,
            is_demo: bool = True,
            **kwargs
    ) -> PipeOutput:
        assert self.nodes, PipelineError('You have to have at least one node to call a pipeline.')

        for node_name in self.nodes:
            data = self._call_node(node_name, data, is_demo=is_demo, **kwargs)

        data = self.modify_output(
            data, self.storage.retriever_docs_native,
            self.storage.retriever_docs_translated, return_translated
        )

        for item in data:
            item['output']['answers'] = [answer for answer in item['output']['answers'] if answer['total_score'] > threshold]

        return data

    def add_node(self, node: Any, name: str, is_technical: bool = False, demo_only: bool = False, **kwargs):
        pipe_type = node.pipe_type

        assert name not in self.nodes, PipelineError(
            f'A node with a name {name} ({pipe_type} pipeline type) is already exists in this pipeline.'
        )

        kwargs = self._get_update_texts_kwargs(pipe_type, kwargs)

        self.nodes[name] = {
            'node': node(name=name, **kwargs),
            'is_technical': is_technical,
            'demo_only': demo_only
        }

    def fit(
            self,
            val_size: float = 0.2,
            batch_size: int = 1,
            top_n_errors: Union[int, List[int]] = [1, 3, 5, 10],
            evaluate: bool = True,
            eval_step: int = 5
    ):
        if not val_size:
            evaluate = False

        if not evaluate:
            top_n_errors = []
            val_size = 0

        trainable_nodes = [node_name for node_name in self.nodes if not self.nodes[node_name]['is_technical']]
        assert trainable_nodes, PipelineError('None of this pipeline nodes are trainable')

        self.storage.make_data_loaders(val_size=val_size, batch_size=batch_size)

        assert self.storage.train_loader, PipelineError('No train data is available')

        train_previous_outputs = []
        for batch in self.storage.train_loader:
            train_previous_outputs.extend(self.add_standard_answers(
                self.standardize_input(batch['question']),
                len(self.storage.retriever_docs_native)
            ))

        val_previous_outputs = []
        for batch in self.storage.val_loader:
            val_previous_outputs.extend(self.add_standard_answers(
                self.standardize_input(batch['question']),
                len(self.storage.retriever_docs_native)
            ))

        for node_name in trainable_nodes:
            item = self.nodes[node_name]
            node = item['node']

            if node.pipe_type in ['retriever', 'ranker']:
                node.fit(
                    self.storage.train_loader, self.storage.val_loader, train_previous_outputs, val_previous_outputs,
                    self.storage.retriever_docs_native, self.storage.retriever_docs_translated,
                    top_n_errors=top_n_errors, node=node if evaluate else None,
                    eval_step=eval_step, storage_path=self.storage.storage_path
                )

            elif node.pipe_type == 'catboost':
                node.fit(
                    self.storage.train_loader+self.storage.val_loader,
                    train_previous_outputs, val_previous_outputs,
                    top_n_errors=top_n_errors, storage_path=self.storage.storage_path
                )

            if node_name != trainable_nodes[-1]:
                train_previous_outputs = self._call_node(node_name, train_previous_outputs, is_demo=False)
                if evaluate:
                    val_previous_outputs = self._call_node(node_name, val_previous_outputs, is_demo=False)

        self.run_benchmarks()

    def run_benchmarks(self, top_n_errors: Union[int, List[int]] = [1, 3, 5, 10]):
        test_questions = [item['native_question'][0] for item in self.storage.test_loader]
        test_contexts = [item['native_context'][0] for item in self.storage.test_loader]

        if test_questions:
            pred_contexts = self.__call__(test_questions, threshold=0)
            test_top_n_errors = top_n_qa_error(test_contexts, pred_contexts, top_n_errors)

            with open(f'{self.storage.storage_path}/test_history/test_fitting_results.json', 'w') as w:
                w.write(json.dumps({
                    'test_top_n_errors_history': test_top_n_errors,
                }))

    def add_documents(self, docs_links: list):
        """
        Add and preprocess new documents to the storage
        :param docs_links: links to the documents
        """
        self.storage.add_documents(docs_links)
        self.update_pipeline_texts()

    def del_document(self, doc_name: str):
        """
        Delete a document from the storage by name
        :param doc_name: document name in the storage
        """
        self.storage.del_document(doc_name)
        self.update_pipeline_texts()

    def update_pipeline_texts(self):
        trainable_nodes = [node_name for node_name in self.nodes if not self.nodes[node_name]['is_technical']]
        for node_name in trainable_nodes:
            item = self.nodes[node_name]
            node = item['node']
            kwargs = self._get_update_texts_kwargs(node.pipe_type)
            node._update_texts(**kwargs)

            if node.pipe_type == 'retriever':
                node.embeddings = node.encode(kwargs['texts'])

    def _call_node(self, node_name: str, data: PipeOutput, is_demo: bool = True, **kwargs):
        demo_only = self.nodes[node_name]['demo_only']
        
        if is_demo or (not is_demo and not demo_only):
            node = self.nodes[node_name]['node']
            return node(data) if node_name not in kwargs else node(data, *kwargs[node_name])
        
        return data
