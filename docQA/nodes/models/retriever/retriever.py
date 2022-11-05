from docQA.nodes.models import BaseSentenceSimilarityEmbeddingsModel


class RetrieverEmbeddingsModel(BaseSentenceSimilarityEmbeddingsModel):
    def __init__(
            self,
            model=None,
            optimizer=None,
            loss_func=None,
            config_path='docQA/configs/retriever_config.json',
            name='retriever',
            state=None,
    ):
        super().__init__(model, optimizer, loss_func, config_path, name, state)
