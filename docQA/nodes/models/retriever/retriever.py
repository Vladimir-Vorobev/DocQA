from docQA.nodes.models import BaseSentenceSimilarityEmbeddingsModel


class RetrieverEmbeddingsModel(BaseSentenceSimilarityEmbeddingsModel):
    def __init__(
            self,
            optimizer=None,
            loss_func=None,
            config_path='docQA/configs/retriever_config.json',
    ):
        super().__init__(optimizer, loss_func, config_path, 'retriever')
