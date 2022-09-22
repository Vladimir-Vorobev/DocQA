from docQA.nodes.file_preprocessor import DocProcessor
from docQA.typing_schemas import PipeOutput, TrainData
from docQA.errors import PipelineError

from typing import List, Any


class Pipeline:
    def __init__(
            self,
            docs_links: List[str],
            doc_processor_config_path: str = 'docQA/configs/processor_config.json'
    ):
        self.preprocessor = DocProcessor(docs_links, doc_processor_config_path)
        self.nodes = {}
        self.pipe_types = {}

    def __call__(self, data: Any, **kwargs) -> PipeOutput:
        if not self.nodes:
            raise PipelineError('You have to have at least one node to call a pipeline.')

        for node_name in self.nodes:
            node = self.nodes[node_name]['node']
            data = node(data) if node_name not in kwargs else node(data, *kwargs[node_name])

        data = self.modify_output(data)

        return data

    def add_node(self, node: Any, name: str, is_technical: bool = False, **kwargs):
        assert name not in self.nodes, PipelineError(
            f'A node with a name {name} ({node.pipe_type} pipeline type) is already exists in this pipeline.'
        )

        pipe_type = node.pipe_type
        if pipe_type not in self.pipe_types:
            self.pipe_types[pipe_type] = 1
        else:
            self.pipe_types[pipe_type] += 1

        if 'number' not in kwargs:
            kwargs['number'] = self.pipe_types[pipe_type] - 1

        if pipe_type == 'retriever':
            if self.preprocessor.retriever_docs_translated:
                kwargs['texts'] = self.preprocessor.retriever_docs_translated
            else:
                kwargs['texts'] = self.preprocessor.retriever_docs_native

        elif pipe_type == 'ranker':
            if self.preprocessor.ranker_docs_translated:
                kwargs['texts'] = self.preprocessor.ranker_docs_translated
            else:
                kwargs['texts'] = self.preprocessor.ranker_docs_native

        self.nodes[name] = {
            'node': node(**kwargs),
            'is_technical': is_technical
        }

    def fit(self, data: TrainData):
        for node_name in self.nodes:
            item = self.nodes[node_name]
            node = item['node']
            is_technical = item['is_technical']
            if not is_technical:
                node.config.is_training = True
                node.fine_tune(data)
                node.config.is_training = False

    def modify_output(self, data):
        if isinstance(data, dict):
            return data

        for item in data:
            for answer_index in range(len(item['output']['answers'])):
                answer = item['output']['answers'][answer_index]
                answer['total_score'] /= answer['weights_sum'] if answer['weights_sum'] else 1
                new_answer = {'answer': self.preprocessor.retriever_docs_native[answer['index']]}
                del answer['index'], answer['weights_sum']
                new_answer.update(answer)
                item['output']['answers'][answer_index] = new_answer

        return data
