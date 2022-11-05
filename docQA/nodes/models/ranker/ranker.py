from docQA.nodes.models import BaseSentenceSimilarityEmbeddingsModel


class RankerEmbeddingsModel(BaseSentenceSimilarityEmbeddingsModel):
    def __init__(
            self,
            model=None,
            optimizer=None,
            loss_func=None,
            config_path='docQA/configs/ranker_config.json',
            name='ranker',
            state=None
    ):
        super().__init__(model, optimizer, loss_func, config_path, name, state)
