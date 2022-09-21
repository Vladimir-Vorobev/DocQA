from docQA.errors.base import BaseError


class PipelineError(BaseError):
    def __init__(self, message):
        super().__init__(message)
