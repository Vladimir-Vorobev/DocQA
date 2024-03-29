from docQA.typing_schemas import PipeOutput
from docQA.pipelines.base import BasePipeline
from docQA.nodes.translator import Translator

from typing import List, Union


class TranslatorPipeline(BasePipeline, Translator):
    pipe_type = 'translator'

    def __init__(
            self,
            model_name: str = 'facebook/wmt19-ru-en',
            max_length: int = 512,
            num_beams: int = 5,
            batch_size: int = 8,
            device: str = 'cuda',
            name: str = 'translator'
    ):
        BasePipeline.__init__(self)
        Translator.__init__(self, model_name, max_length, num_beams, batch_size, device)

    def __call__(
            self,
            data: Union[str, List[str]],
            standardized: bool = True
    ) -> PipeOutput:

        if standardized:
            data = self.standardize_input(data)
        elif isinstance(data, str):
            data = [data]

        for index in range(len(data)):
            if standardized:
                data[index]['modified_input'] = self._translate(data[index]['modified_input'])
            else:
                data[index] = self._translate(data[index])

        return data
