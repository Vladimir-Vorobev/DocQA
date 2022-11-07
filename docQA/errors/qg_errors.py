from docQA.errors.base import BaseError


class QgError(BaseError):
    def __init__(self, message):
        super().__init__(message)
