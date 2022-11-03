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

        for link in tqdm(docs_links, desc='Opening docs'):
            with open(link, encoding=config.doc_encoding) as r:
                doc = r.read()
                doc = [text for text in doc.split(self.retriever_sep) if text.strip()]
                if not self.replace_retriever_sep:
                    doc = [self.retriever_sep + text for text in doc]

                docs.extend(doc)

        all_docs = docs.copy()
        docs = list(set(docs) - set(old_docs))
        if not docs:
            return

        self.retriever_docs_native.extend(docs)

        self.ranker_docs_native.extend(
            [self._create_ranker_doc(doc) for doc in docs]
        )

        if self.translator:
            self.retriever_docs_translated.extend(
                [self.translator._translate(doc) for doc in tqdm(docs, desc='Translating docs paragraphs')]
            )

            for doc in tqdm(self.ranker_docs_native, desc='Grouping and translating docs by paragraphs'):
                translated_doc = []

                for text in doc:
                    translated_doc.append(self.translator._translate(text))

                self.ranker_docs_translated.append(translated_doc)

        docs.extend(old_docs)

        with open(config.docs_file_path, 'w') as w:
            w.write(json.dumps({
                'docs': all_docs,
                'retriever_docs_native': self.retriever_docs_native,
                'retriever_docs_translated': self.retriever_docs_translated,
                'ranker_docs_native': self.ranker_docs_native,
                'ranker_docs_translated': self.ranker_docs_translated,
            }))

    def _create_ranker_doc(self, doc):
        if self.ranker_sep:
            return [text for text in doc.split(self.ranker_sep) if text]
        else:
            return [doc]
