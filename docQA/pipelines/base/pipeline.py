from docQA.nodes.file_preprocessor import DocProcessor
from docQA.typing_schemas import PipeOutput, TrainData
from docQA.errors import PipelineError
from docQA.utils.torch import BaseDataset
from docQA import seed

from typing import Union, List, Any
import torch
import pickle


class Pipeline:
    def __init__(
            self,
            docs_links: List[str] = [],
            doc_processor_config_path: str = 'docQA/configs/processor_config.json',
    ):
        self.preprocessor = DocProcessor(docs_links, doc_processor_config_path)
        self.nodes = {}

    def __getstate__(self) -> dict:
        return {
            'preprocessor': self.preprocessor,
            'nodes': self.nodes
        }

    def __setstate__(self, state: dict):
        self.preprocessor = state['preprocessor']
        self.nodes = state['nodes']

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
            data = self.modify_output(data, return_translated)

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

        train_previous_outputs = [item['question'] for item in train_dataset]
        val_previous_outputs = [item['question'] for item in val_dataset]

        for node_name in self.nodes:
            item = self.nodes[node_name]
            node = item['node']
            is_technical = item['is_technical']

            if not is_technical:
                fit_pipe = Pipeline() if evaluate else None

                if fit_pipe:
                    fit_pipe.preprocessor = self.preprocessor

                    for fit_node_name in [name for name in self.nodes if not self.nodes[name]['demo_only']]:
                        fit_pipe.nodes[fit_node_name] = self.nodes[fit_node_name]
                        if fit_node_name == node_name:
                            break

                if node.pipe_type in ['retriever', 'ranker']:
                    node.fit(train_dataset, val_dataset, top_n_errors=top_n_errors, pipe=fit_pipe, eval_step=eval_step)

                elif node.pipe_type == 'catboost':
                    node.fit(
                        data, train_previous_outputs, val_previous_outputs, top_n_errors=top_n_errors, pipe=fit_pipe
                    )

            train_previous_outputs = self._call_node(node_name, train_previous_outputs, is_demo=False)
            val_previous_outputs = self._call_node(node_name, val_previous_outputs, is_demo=False)

    def modify_output(self, data, return_translated=False):
        if isinstance(data, dict):
            return data

        for item in data:
            for answer_index in range(len(item['output']['answers'])):
                answer = item['output']['answers'][answer_index]
                answer['total_score'] /= answer['weights_sum'] if answer['weights_sum'] else 1
                new_answer = {'answer': self.preprocessor.retriever_docs_native[answer['index']]}

                if return_translated and self.preprocessor.retriever_docs_translated:
                    new_answer['translated_answer'] = self.preprocessor.retriever_docs_translated[answer['index']]

                del answer['index'], answer['weights_sum']
                new_answer.update(answer)
                item['output']['answers'][answer_index] = new_answer

        return data

    def _call_node(self, node_name, data, is_demo=True, **kwargs):
        demo_only = self.nodes[node_name]['demo_only']
        
        if is_demo or (not is_demo and not demo_only):
            node = self.nodes[node_name]['node']
            return node(data) if node_name not in kwargs else node(data, *kwargs[node_name])
        
        return data

    def save(self, fine_name: str = 'pipeline.pkl'):
        b = pickle.dumps(self)
        k = pickle.loads(b)
        print(k)

        with open(fine_name, 'wb') as w:
            pickle.dump({'nodes': getattr(self, 'nodes'), 'preprocessor': getattr(self, 'preprocessor')}, w)

    def load(self, fine_name):
        with open(fine_name, 'rb') as r:
            k = pickle.load(r)
