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

        if pipe_type in ['retriever', 'catboost']:
            if self.storage.retriever_docs_translated:
                kwargs['texts'] = self.storage.retriever_docs_translated
            else:
                kwargs['texts'] = self.storage.retriever_docs_native

        if pipe_type == 'ranker':
            if self.storage.ranker_docs_translated:
                kwargs['texts'] = self.storage.ranker_docs_translated
            else:
                kwargs['texts'] = self.storage.ranker_docs_native

        if pipe_type == 'catboost':
            kwargs['native_texts'] = self.storage.retriever_docs_native

        self.nodes[name] = {
            'node': node(name=name, **kwargs),
            'is_technical': is_technical,
            'demo_only': demo_only
        }

    def fit(self, val_size: float = 0.2, top_n_errors: Union[int, List[int]] = [1, 3, 5, 10], evaluate: bool = True, eval_step: int = 5):
        if not evaluate:
            top_n_errors = []

        trainable_nodes = [node_name for node_name in self.nodes if not self.nodes[node_name]['is_technical']]
        assert trainable_nodes, PipelineError('None of this pipeline nodes are trainable')

        self.storage.make_data_loaders(val_size=val_size)

        assert self.storage.train_loader, PipelineError('No train data is available')

        # после этого сломается при ranker_sep != ''
        train_previous_outputs = self.add_standard_answers(
            self.standardize_input([item['question'] for item in self.storage.train_loader]),
            len(self.storage.retriever_docs_native)
        )

        val_previous_outputs = self.add_standard_answers(
            self.standardize_input([item['question'] for item in self.storage.val_loader]),
            len(self.storage.retriever_docs_native)
        ) if self.storage.val_loader else []

        test_questions = [item['native_question'] for item in self.storage.test_loader]
        test_contexts = [item['native_context'] for item in self.storage.test_loader]

        for node_name in trainable_nodes:
            item = self.nodes[node_name]
            node = item['node']

            if node.pipe_type in ['retriever', 'ranker']:
                node.fit(
                    self.storage.train_loader, self.storage.val_loader, train_previous_outputs, val_previous_outputs,
                    self.storage.retriever_docs_native, self.storage.retriever_docs_translated,
                    top_n_errors=top_n_errors, node=node if evaluate else None, eval_step=eval_step, storage_path=self.storage.storage_path
                )

            elif node.pipe_type == 'catboost':
                node.fit(
                    self.storage.train_loader+self.storage.val_loader,
                    train_previous_outputs, val_previous_outputs,
                    top_n_errors=top_n_errors, storage_path=self.storage.storage_path
                )

            if node_name != trainable_nodes[-1]:
                train_previous_outputs = self._call_node(node_name, train_previous_outputs, is_demo=False)
                val_previous_outputs = self._call_node(node_name, val_previous_outputs, is_demo=False)

        if test_questions:
            pred_contexts = self.__call__(test_questions, threshold=0)
            test_top_n_errors = top_n_qa_error(test_contexts, pred_contexts, top_n_errors)

            with open(f'{self.storage.storage_path}/test_history/test_fitting_results.json', 'w') as w:
                w.write(json.dumps({
                    'test_top_n_errors_history': test_top_n_errors,
                }))

    def _call_node(self, node_name, data, is_demo=True, **kwargs):
        demo_only = self.nodes[node_name]['demo_only']
        
        if is_demo or (not is_demo and not demo_only):
            node = self.nodes[node_name]['node']
            return node(data) if node_name not in kwargs else node(data, *kwargs[node_name])
        
        return data
