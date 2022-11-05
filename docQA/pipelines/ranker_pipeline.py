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
            model: str = None,
            optimizer: Any = None,
            loss_func: Any = None,
            weight: float = 1.0,
            name: str = 'ranker',
            return_num: int = 10,
            config_path: str = 'docQA/configs/ranker_config.json'
    ):
        state = {
            'texts': texts,
            'model': model,
            'optimizer': optimizer,
            'loss_func': loss_func,
            'weight': weight,
            'return_num': return_num,
            'config_path': config_path
        }

        BasePipeline.__init__(self)
        RankerEmbeddingsModel.__init__(self, model, optimizer, loss_func, config_path, name, state)
        self.texts = texts
        self.weight = weight
        self.return_num = return_num

    def __call__(
            self,
            data: Union[str, List[str]],
            return_num: int = 10
    ) -> PipeOutput:
        if self.return_num != return_num and return_num == 10:
            return_num = self.return_num

        data = self.standardize_input(data)
        data = self.add_standard_answers(data, len(self.texts))

        if return_num == -1:
            return_num = len(data)

        for index in range(len(data)):
            answers = data[index]['output']['answers']

            embeddings = self.encode(
                [data[index]['modified_input'], *[self.texts[answer['index']][0] for answer in answers]]
            )

            for answer_index, embedding in zip(range(len(answers)), embeddings[1:]):
                answer = answers[answer_index]

                score = cosine_similarity(embeddings[0], embedding) * self.weight

                answer['scores'][f'{self.name}_cos_sim'] = score
                answer['total_score'] += score
                answer['weights_sum'] += self.weight

            data[index]['output']['answers'] = \
                sorted(answers, key=lambda x: x['total_score'], reverse=True)[:return_num]

        return data
