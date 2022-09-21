from docQA.typing_schemas import PipeOutput
from docQA.errors import PipelineError

from typing import List, Union, Any


class BasePipeline:
    pipe_type = 'base'

    def __init__(self):
        self.number = 0
        self.name = ''

    def __call__(self, data: Any) -> PipeOutput:
        raise PipelineError(f'Call method is not supported in pipelines with a {self.pipe_type} type.')

    @staticmethod
    def standardize_input(data: Union[str, List[str], PipeOutput]) -> PipeOutput:
        if isinstance(data, str):
            return [{'input': data, 'output': {'answers': []}, 'modified_input': data}]
        elif isinstance(data[0], str):
            return [{'input': item, 'output': {'answers': []}, 'modified_input': item} for item in data]
        else:
            return data

    @staticmethod
    def add_standard_answers(data: PipeOutput, answer_len: int):
        if data[0]['output']['answers']:
            return data

        answers = [{'index': index, 'total_score': 0, 'weights_sum': 0, 'scores': {}} for index in range(answer_len)]

        for index in range(len(data)):
            data[index]['output']['answers'] = answers

        return data
