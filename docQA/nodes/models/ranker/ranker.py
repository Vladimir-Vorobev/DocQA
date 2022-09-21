from docQA.nodes.models import BaseSentenceSimilarityEmbeddingsModel


class RankerEmbeddingsModel(BaseSentenceSimilarityEmbeddingsModel):
    def __init__(
            self,
            optimizer=None,
            loss_func=None,
            config_path='docQA/configs/ranker_config.json',
    ):
        super().__init__(optimizer, loss_func, config_path, 'ranker')
