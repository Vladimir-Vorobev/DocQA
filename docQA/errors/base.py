class AssertionError(AssertionError):
    """
    Re-initialization of the AssertionError class with the purpose of calling
    specific errors instead of assertion errors while using assert construction
    """
    def __init__(self, error):
        raise error


class BaseError(Exception):
    """
    A base parental class for each error class in DocQA
    """
    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return self.message

    def __str__(self):
        return self.message
