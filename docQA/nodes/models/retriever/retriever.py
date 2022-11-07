from docQA.nodes.models import BaseSentenceSimilarityEmbeddingsModel

from typing import Any


class RetrieverEmbeddingsModel(BaseSentenceSimilarityEmbeddingsModel):
    def __init__(
            self,
            model: Any = None,
            optimizer: Any = None,
            loss_func: Any = None,
            config_path: str = 'docQA/configs/retriever_config.json',
            name: str = 'retriever',
            state: dict = None,
    ):
        super().__init__(model, optimizer, loss_func, config_path, name, state)
