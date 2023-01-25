from docQA.mixins import ModifyOutputMixin
from docQA.typing_schemas import PipeOutput
from docQA.errors import PipelineError

from typing import List, Union, Any
from copy import deepcopy
import pickle


class BasePipeline(ModifyOutputMixin):
    pipe_type = 'base'

    def __init__(self):
        self.name = ''

    def __call__(self, data: Any) -> PipeOutput:
        raise PipelineError(f'Call method is not supported in pipelines with a {self.pipe_type} type.')

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def standardize_input(data: Union[str, List[str], PipeOutput]) -> PipeOutput:
        if isinstance(data, str):
            return [{'input': data, 'output': {'answers': []}, 'modified_input': data}]
        elif isinstance(data[0], str):
            return [{'input': item, 'output': {'answers': []}, 'modified_input': item} for item in data]
        elif isinstance(data[0], list):
            return [{'input': item[0], 'output': {'answers': []}, 'modified_input': item[0]} for item in data]
        else:
            return data

    @staticmethod
    def add_standard_answers(data: PipeOutput, answer_len: int):
        if data[0]['output']['answers']:
            return data

        answers = [{'index': index, 'total_score': 0, 'weights_sum': 0, 'scores': {}} for index in range(answer_len)]

        for index in range(len(data)):
            data[index]['output']['answers'] = deepcopy(answers)

        return data

    def save(self, file_name: str = 'pipeline.pkl'):
        with open(file_name, 'wb') as w:
            pickle.dump(self.__getstate__(), w)

    def load(self, file_name: str):
        with open(file_name, 'rb') as r:
            self.__setstate__(pickle.load(r))

    def _get_update_texts_kwargs(self, pipe_type: str, kwargs: dict = {}):
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

        return kwargs

    def _update_texts(self, **kwargs):
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
