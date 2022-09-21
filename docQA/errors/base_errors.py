from docQA.errors.base import BaseError


class ConfigError(BaseError):
    def __init__(self, param, config_path):
        super().__init__(f'Config "{config_path}" does not contain "{param}" param.')


class DeviceError(BaseError):
    def __init__(self, device):
        super().__init__(f'Device "{device}" is not supported now.')


class SentenceEmbeddingsModelError(BaseError):
    def __init__(self, error_type):
        if error_type == 'training':
            super().__init__(
                'The model is not in the training mode right now. Please, set config["is_training"] = True before training.'
            )
        elif error_type == 'evaluating':
            super().__init__(
                'The model is not in the evaluation mode right now. Please, set config["is_training"] = False before encoding.'
            )


class DocQAInputError(BaseError):
    def __init__(self, input_type):
        super().__init__(f'The input should be a list of strings or a string, not {input_type}.')
