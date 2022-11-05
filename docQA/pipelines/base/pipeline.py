from .base_pipeline import BasePipeline
from docQA.nodes.file_preprocessor import DocProcessor
from docQA.typing_schemas import PipeOutput, TrainData
from docQA.errors import PipelineError
from docQA.utils.torch import BaseDataset
from docQA import seed

from typing import Union, List, Any
import torch
import pickle


class Pipeline(BasePipeline):
    def __init__(
            self,
            docs_links: List[str] = [],
            doc_processor_config_path: str = 'docQA/configs/processor_config.json',
    ):
        super().__init__()
        self.preprocessor = DocProcessor(docs_links, doc_processor_config_path)
        self.nodes = {}

    def __call__(
            self,
            data: Any,
            return_translated: bool = False,
            threshold: float = 0.3,
            return_output: bool = True,
            is_demo: bool = True,
            **kwargs
    ) -> PipeOutput:
        if not self.nodes:
            raise PipelineError('You have to have at least one node to call a pipeline.')
        for node_name in self.nodes:
            data = self._call_node(node_name, data, is_demo=is_demo, **kwargs)
        if return_output:
            data = self.modify_output(
                data, self.preprocessor.retriever_docs_native,
                self.preprocessor.retriever_docs_translated, return_translated
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
            if self.preprocessor.retriever_docs_translated:
                kwargs['texts'] = self.preprocessor.retriever_docs_translated
            else:
                kwargs['texts'] = self.preprocessor.retriever_docs_native

        if pipe_type == 'ranker':
            if self.preprocessor.ranker_docs_translated:
                kwargs['texts'] = self.preprocessor.ranker_docs_translated
            else:
                kwargs['texts'] = self.preprocessor.ranker_docs_native

        if pipe_type == 'catboost':
            kwargs['native_texts'] = self.preprocessor.retriever_docs_native

        self.nodes[name] = {
            'node': node(name=name, **kwargs),
            'is_technical': is_technical,
            'demo_only': demo_only
        }

    def fit(self, data: TrainData, val_size: float = 0.2, top_n_errors: Union[int, List[int]] = [1, 3, 5, 10], evaluate: bool = True, eval_step: int = 5):
        if not evaluate:
            top_n_errors = []

        dataset = BaseDataset(data)
        train_length = int(len(dataset) * (1 - val_size))
        val_length = len(dataset) - train_length

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_length, val_length], generator=torch.Generator().manual_seed(seed)
        )

        # после этого сломается при ranker_sep != ''
        train_previous_outputs = self.add_standard_answers(
            self.standardize_input([item['question'] for item in train_dataset]),
            len(self.preprocessor.retriever_docs_native)
        )

        val_previous_outputs = self.add_standard_answers(
            self.standardize_input([item['question'] for item in val_dataset]),
            len(self.preprocessor.retriever_docs_native)
        )

        trainable_nodes = [node_name for node_name in self.nodes if not self.nodes[node_name]['is_technical']]

        # добавить ошибку
        assert trainable_nodes

        for node_name in trainable_nodes:
            item = self.nodes[node_name]
            node = item['node']

            if node.pipe_type in ['retriever', 'ranker']:
                node.fit(
                    train_dataset, val_dataset, train_previous_outputs, val_previous_outputs,
                    self.preprocessor.retriever_docs_native, self.preprocessor.retriever_docs_translated,
                    top_n_errors=top_n_errors, node=node if evaluate else None, eval_step=eval_step
                )

            elif node.pipe_type == 'catboost':
                node.fit(
                    data.copy(), train_previous_outputs, val_previous_outputs,
                    top_n_errors=top_n_errors
                )

            if node_name != trainable_nodes[-1]:
                train_previous_outputs = self._call_node(node_name, train_previous_outputs, is_demo=False)
                val_previous_outputs = self._call_node(node_name, val_previous_outputs, is_demo=False)

    def _call_node(self, node_name, data, is_demo=True, **kwargs):
        demo_only = self.nodes[node_name]['demo_only']
        
        if is_demo or (not is_demo and not demo_only):
            node = self.nodes[node_name]['node']
            return node(data) if node_name not in kwargs else node(data, *kwargs[node_name])
        
        return data

    def save(self, fine_name: str = 'pipeline.pkl'):
        with open(fine_name, 'wb') as w:
            pickle.dump(self, w)

    def load(self, fine_name):
        with open(fine_name, 'rb') as r:
            self.__dict__ = pickle.load(r).__dict__
