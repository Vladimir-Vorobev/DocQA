from docQA.typing_schemas import PipeOutput
from docQA.pipelines.base import BasePipeline
from docQA.nodes.models import RetrieverEmbeddingsModel
from docQA.metrics import cosine_similarity

from typing import List, Union, Any


class RetrieverPipeline(BasePipeline, RetrieverEmbeddingsModel):
    pipe_type = 'retriever'

    def __init__(
            self,
            texts: List[str],
            model: str = None,
            optimizer: Any = None,
            loss_func: Any = None,
            weight: float = 1.0,
            name: str = 'retriever',
            return_num: int = 30,
            config_path: str = 'docQA/configs/retriever_config.json'
    ):
        BasePipeline.__init__(self)
        RetrieverEmbeddingsModel.__init__(self, model, optimizer, loss_func, config_path, name)
        self.texts = texts
        self.embeddings = self.encode(texts)
        self.weight = weight
        self.return_num = return_num

    def __getstate__(self) -> dict:
        return {
            'texts': self.texts,
            'embeddings': self.embeddings,
            'name': self.name,
            'config': self.config,
            'model': self.model,
            'tokenizer': self.tokenizer,
            'optimizer': self.optimizer,
            'loss_func': self.loss_func,
            'autocast_type': self.autocast_type,
        }

    def __setstate__(self, state: dict):
        self.texts = state['texts']
        self.embeddings = state['embeddings']
        self.name = state['name']
        self.config = state['config']
        self.model = state['model']
        self.tokenizer = state['tokenizer']
        self.optimizer = state['optimizer']
        self.loss_func = state['loss_func']
        self.autocast_type = state['autocast_type']

    def __call__(
            self,
            data: Union[str, List[str]],
            return_num: int = 30
    ) -> PipeOutput:
        if self.return_num != return_num and return_num == 30:
            return_num = self.return_num

        data = self.standardize_input(data)
        data = self.add_standard_answers(data, len(self.texts))

        if return_num == -1:
            return_num = len(data)

        data_embeddings = self.encode([item['modified_input'] for item in data])

        for index, embedding in zip(range(len(data)), data_embeddings):
            answers = data[index]['output']['answers']
            for answer_index in range(len(answers)):
                answer = answers[answer_index]

                score = cosine_similarity(embedding, self.embeddings[answer['index']]) * self.weight

                answer['scores'][f'{self.name}_cos_sim'] = score
                answer['total_score'] += score
                answer['weights_sum'] += self.weight

            data[index]['output']['answers'] = \
                sorted(answers, key=lambda x: x['total_score'], reverse=True)[:return_num]

        return data
