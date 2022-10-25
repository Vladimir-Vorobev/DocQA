from docQA.configs import ConfigParser
from docQA.nodes.translator import Translator

from tqdm.autonotebook import tqdm
import os
import json


class DocProcessor:
    def __init__(
            self,
            docs_links,
            config_path='docQA/configs/processor_config.json',
    ):
        config = ConfigParser(config_path)

        self.retriever_sep = config.retriever_sep
        self.ranker_sep = config.ranker_sep
        self.replace_retriever_sep = config.replace_retriever_sep
        self.native_lang = config.native_lang
        self.retriever_docs_native = []
        self.retriever_docs_translated = []
        self.ranker_docs_native = []
        self.ranker_docs_translated = []
        self.translator = Translator(config.model_name, device=config.device) if config.model_name else None
        old_docs = []
        docs = []

        if os.path.isfile(config.docs_file_path):
            with open(config.docs_file_path) as r:
                docs_file = json.load(r)
                old_docs = docs_file['docs']
                self.retriever_docs_native = docs_file['retriever_docs_native']
                self.retriever_docs_translated = docs_file['retriever_docs_translated']
                self.ranker_docs_native = docs_file['ranker_docs_native']
                self.ranker_docs_translated = docs_file['ranker_docs_translated']

        for link in tqdm(docs_links, ascii=True, desc='Opening docs'):
            doc = [text for text in open(link, encoding=config.doc_encoding).readlines() if text]
            doc = ''.join(doc)
            doc = [text for text in doc.split(self.retriever_sep) if text]
            if not self.replace_retriever_sep:
                doc = [self.retriever_sep + text for text in doc]

            docs.extend(doc)

        docs = list(set(docs) - set(old_docs))
        if not docs:
            return

        self.retriever_docs_native.extend(
            [doc.replace('\n', '') for doc in docs if doc != '\n']
        )

        self.ranker_docs_native.extend(
            [self._create_ranker_doc(doc) for doc in tqdm(docs, ascii=True, desc='Grouping docs by paragraphs')]
        )

        if self.translator:
            self.retriever_docs_translated.extend(
                [doc for doc in tqdm(
                    self.translator._translate('\n'.join(docs)).split('\n'),
                    ascii=True, desc='Translating docs paragraphs'
                )]
            )
            self.ranker_docs_translated.extend(
                [self._create_ranker_doc(doc) for doc in
                 tqdm(self.retriever_docs_translated, ascii=True, desc='Grouping translated docs by paragraphs')]
            )

        docs.extend(old_docs)

        with open(config.docs_file_path, 'w') as w:
            w.write(json.dumps({
                'docs': docs,
                'retriever_docs_native': self._clean_docs(self.retriever_docs_native),
                'retriever_docs_translated': self._clean_docs(self.retriever_docs_translated),
                'ranker_docs_native': self._clean_docs(self.ranker_docs_native, retriever=False),
                'ranker_docs_translated': self._clean_docs(self.ranker_docs_translated, retriever=False),
            }))

    def _create_ranker_doc(self, doc):
        return [text for text in list(map(lambda x: x.replace('\n', ''), doc.split(self.ranker_sep))) if text]

    # костыль
    @staticmethod
    def _clean_docs(docs, retriever=True):
        if retriever:
            return [doc for doc in docs if doc]

        for i in range(len(docs)):
            docs[i] = [text for text in docs[i] if text]

        return docs

