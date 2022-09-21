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
            optimizer: Any = None,
            loss_func: Any = None,
            weight: float = 1.0,
            number: int = 0,
            config_path: str = 'docQA/configs/retriever_config.json'
    ):
        BasePipeline.__init__(self)
        RetrieverEmbeddingsModel.__init__(self, optimizer, loss_func, config_path)
        self.texts = texts
        self.embeddings = self.encode(texts)
        self.weight = weight
        self.number = number

    def __call__(
            self,
            data: Union[str, List[str]],
            retriever_n: int = 30
    ) -> PipeOutput:
        data = self.standardize_input(data)
        data = self.add_standard_answers(data, len(self.texts))
        data_embeddings = self.encode([item['modified_input'] for item in data])

        for index, embedding in zip(range(len(data)), data_embeddings):
            answers = data[index]['output']['answers']
            for answer_index in range(len(answers)):
                answer = answers[answer_index]

                score = cosine_similarity(embedding, self.embeddings[answer['index']]) * self.weight

                answer['scores'][f'retriever_{self.number}_cos_sim'] = score
                answer['total_score'] += score
                answer['weights_sum'] += self.weight

            data[index]['output']['answers'] = \
                sorted(answers, key=lambda x: x['total_score'], reverse=True)[:retriever_n]

        return data
