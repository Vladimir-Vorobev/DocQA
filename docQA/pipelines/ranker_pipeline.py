from docQA.typing_schemas import PipeOutput
from docQA.pipelines.base import BasePipeline
from docQA.nodes.models import RankerEmbeddingsModel
from docQA.metrics import cosine_similarity

from typing import List, Union, Any


class RankerPipeline(BasePipeline, RankerEmbeddingsModel):
    pipe_type = 'ranker'

    def __init__(
            self,
            texts: List[str],
            optimizer: Any = None,
            loss_func: Any = None,
            weight: float = 1.0,
            number: int = 0,
            config_path: str = 'docQA/configs/ranker_config.json'
    ):
        BasePipeline.__init__(self)
        RankerEmbeddingsModel.__init__(self, optimizer, loss_func, config_path)
        self.texts = texts
        self.weight = weight
        self.number = number

    def __call__(
            self,
            data: Union[str, List[str]],
            ranker_n: int = 10
    ) -> PipeOutput:
        data = self.standardize_input(data)
        data = self.add_standard_answers(data, len(self.texts))

        for index in range(len(data)):
            answers = data[index]['output']['answers']
            # will lead to a bug [data[index]['modified_input'], *[self.texts[answer['index']][0] for answer in answers
            embeddings = self.encode(
                [data[index]['modified_input'], *[self.texts[answer['index']][0] for answer in answers if self.texts[answer['index']]]]
            )

            for answer_index, embedding in zip(range(len(answers)), embeddings[1:]):
                answer = answers[answer_index]

                score = cosine_similarity(embeddings[0], embedding) * self.weight

                answer['scores'][f'ranker_{self.number}_cos_sim'] = score
                answer['total_score'] += score
                answer['weights_sum'] += self.weight

            data[index]['output']['answers'] = \
                sorted(answers, key=lambda x: x['total_score'], reverse=True)[:ranker_n]

        return data
