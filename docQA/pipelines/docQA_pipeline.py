from docQA.nodes.file_preprocessor import DocProcessor
from docQA.utils import copy_class_parameters
from docQA.metrics import cosine_similarity
from docQA.nodes.translator import translator
from docQA.errors import DocQAInputError
from docQA.nodes.models import RetrieverEmbeddingsModel, RankerEmbeddingsModel


class DocQA:
    def __init__(
            self,
            docs_links,
            doc_processor_config_path='../configs/processor_config.json',

            ranker_optimizer=None,
            ranker_loss_func=None,
            ranker_config_path='../configs/ranker_config.json',

            retriever_optimizer=None,
            retriever_loss_func=None,
            retriever_config_path='../configs/retriever_config.json',
    ):
        copy_class_parameters(self, DocProcessor(docs_links, doc_processor_config_path))
        self.ranker_embeddings_model = RankerEmbeddingsModel(ranker_optimizer, ranker_loss_func, ranker_config_path)
        self.retriever_embeddings_model = RetrieverEmbeddingsModel(retriever_optimizer, retriever_loss_func,
                                                                   retriever_config_path)
        self.encode_ranker_docs_embeddings()

    def __call__(self, query, ranker_n=50, retriever_n=10, return_en=False):
        if ranker_n == -1:
            ranker_n = len(self._ranker_docs_embeddings)
        if retriever_n == -1:
            retriever_n = len(self._ranker_docs_embeddings)
        return self._postprocess(self._forward(self._preprocess(query), ranker_n, retriever_n, return_en))

    def encode_ranker_docs_embeddings(self):
        self._ranker_docs_embeddings = self.ranker_embeddings_model.encode(self._ranker_docs_en)

    def _ranker(self, query, ranker_n):
        query_embedding = self.ranker_embeddings_model.encode(query)[0]

        docs_similarity = [
            [cosine_similarity(query_embedding, self._ranker_docs_embeddings[i]), i] for i in
            range(len(self._ranker_docs_embeddings))
        ]

        return sorted(docs_similarity, key=lambda x: -x[0])[:ranker_n]

    def _retriever(self, query, ranker_outputs, retriever_n, return_en):
        retriever_docs_native = []
        retriever_docs_en = query
        for output in ranker_outputs:
            for doc, doc_en in zip(self._retriever_docs[output[1]], self._retriever_docs_en[output[1]]):
                retriever_docs_native.append([doc, output[0]])
                retriever_docs_en.append(doc_en)

        retriever_embeddings = self.retriever_embeddings_model.encode(retriever_docs_en)

        if return_en:
            retriever_docs = retriever_docs_en[1:]
        else:
            retriever_docs = [doc[0] for doc in retriever_docs_native]

        docs_similarity = [
            [retriever_docs[i - 1], retriever_docs_native[i - 1][1],
             cosine_similarity(retriever_embeddings[0], retriever_embeddings[i])] for i in
            range(1, len(retriever_embeddings))
        ]

        return sorted(docs_similarity, key=lambda x: -(x[2] + x[1]))[:retriever_n]

    @staticmethod
    def _preprocess(inputs):
        if isinstance(inputs, str):
            return [translator.translate(inputs)]
        else:
            raise DocQAInputError(type(inputs))

    def _forward(self, model_inputs, ranker_n, retriever_n, return_en):
        ranker_outputs = self._ranker(model_inputs, ranker_n)
        retriever_outputs = self._retriever(model_inputs, ranker_outputs, retriever_n, return_en)
        outputs = retriever_outputs
        return outputs

    @staticmethod
    def _postprocess(model_outputs):
        return [{'answer': output[0], 'retriever_score': output[1], 'ranker_score': output[2]} for output in
                model_outputs]

    def fine_tune(self, ranker_train_data=None, retriever_train_data=None, ranker=True, retriever=True):
        if ranker:
            self.ranker_embeddings_model.config.is_training = True
            self.ranker_embeddings_model.fine_tune(ranker_train_data, pipe=self)
            self.ranker_embeddings_model.config.is_training = False
        if retriever:
            self.retriever_embeddings_model.config.is_training = True
            self.retriever_embeddings_model.fine_tune(retriever_train_data, pipe=self)
            self.retriever_embeddings_model.config.is_training = False
