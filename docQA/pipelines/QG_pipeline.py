from docQA.nodes.models import QuestionGenerator
from docQA.pipelines.base import BasePipeline


class QgPipeline(BasePipeline, QuestionGenerator):
    pipe_type = 'qg'

    def __init__(
            self,
            device: str = 'cuda',
            path_to_save: str = '',
            cdqa_pipe = None,
            name: str = 'qg'
    ):
        self.path_to_save = path_to_save
        BasePipeline.__init__(self)
        QuestionGenerator.__init__(self, cdqa_pipe=cdqa_pipe, device=device)

    def __call__(
            self,
            path_to_save: str,
            standardized: bool = True
    ):
        self._generate_questions(path_to_save=path_to_save)
