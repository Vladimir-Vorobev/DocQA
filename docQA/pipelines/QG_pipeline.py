from docQA.nodes.models import QuestionGenerator
from docQA.pipelines.base import BasePipeline, Pipeline


class QgPipeline(BasePipeline, QuestionGenerator):
    pipe_type = 'qg'

    def __init__(
            self,
            device: str = 'cuda',
            path_to_save: str = '',
            pipe: Pipeline = None,
            name: str = 'qg'
    ):
        self.path_to_save = path_to_save
        BasePipeline.__init__(self)
        QuestionGenerator.__init__(self, pipe=pipe, device=device)

    def __call__(
            self,
            path_to_save: str,
            standardized: bool = True
    ):
        self.generate_questions(path_to_save=path_to_save)
