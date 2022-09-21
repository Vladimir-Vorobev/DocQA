class AssertionError(AssertionError):
    def __init__(self, error):
        raise error


class BaseError(Exception):
    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return self.message

    def __str__(self):
        return self.message
