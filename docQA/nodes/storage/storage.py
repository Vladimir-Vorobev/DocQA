from .doc_converter import DocConverter
from docQA.configs import ConfigParser
from docQA.nodes.translator import Translator
from docQA.utils.torch import BaseDataset

import os
import json
import pandas as pd
import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader


class Storage:
    def __init__(
        self,
        storage_path: str = '',
        storage_name: str = 'base',
        docs_links: list = [],
        config_path='docQA/configs/storage_config.json'
    ):
        self.config = ConfigParser(config_path)
        self.doc_converter = DocConverter
        self.retriever_docs_native = []
        self.retriever_docs_translated = []
        self.ranker_docs_native = []
        self.ranker_docs_translated = []
        self.translator = Translator(self.config.model_name, device=self.config.device) if self.config.model_name else None
        docs = []

        if not storage_path:
            storage_path = os.getcwd()

        self.storage_path = f'{storage_path}/{storage_name}'

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            os.makedirs(f'{self.storage_path}/docs')
            os.makedirs(f'{self.storage_path}/train_history')
            os.makedirs(f'{self.storage_path}/test_history')

        if not os.path.exists(f'{self.storage_path}/storage_data.json'):
            with open(f'{self.storage_path}/storage_data.json', 'w') as w:
                w.write(json.dumps(
                    {
                        'documents': {}, 'datasets': {'train_data': {}, 'benchmark_data': {}},
                        'retriever_range': {}, 'ranker_range': {}
                    }
                ))
        
        with open(f'{self.storage_path}/storage_data.json', 'r') as r:
            files = json.load(r)

        for link in docs_links:
            doc_name = os.path.split(link)[1].replace('.docx', '').replace('.rtf', '').replace('.doc', '')
            if doc_name not in files['documents']:
                new_path = self.doc_converter.convert_doc(link, f'{self.storage_path}/docs/{os.path.split(link)[1]}')
                files['documents'][doc_name] = self._process_doc(new_path)

        files['retriever_range'] = {}
        files['ranker_range'] = {}
        retriever_sum = 0
        ranker_sum = 0

        for doc_name in files['documents']:
            files['retriever_range'][retriever_sum+len(files['documents'][doc_name]['retriever_docs_native'])] = doc_name
            retriever_sum += len(files['documents'][doc_name]['retriever_docs_native'])
            files['ranker_range'][ranker_sum+len(files['documents'][doc_name]['ranker_docs_native'])] = doc_name
            ranker_sum += len(files['documents'][doc_name]['ranker_docs_native'])
        
        with open(f'{self.storage_path}/storage_data.json', 'w') as w:
            json.dump(files, w)

        for name in files['documents']:
            docs.extend(files['documents'][name]['docs'])
            self.retriever_docs_native.extend(files['documents'][name]['retriever_docs_native'])
            self.retriever_docs_translated.extend(files['documents'][name]['retriever_docs_translated'])
            self.ranker_docs_native.extend(files['documents'][name]['ranker_docs_native'])
            self.ranker_docs_translated.extend(files['documents'][name]['ranker_docs_translated']) 

        self.retriever_range = files['retriever_range']
        self.ranker_range = files['ranker_range']

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _process_doc(self, link: str):
        with open(link, encoding=self.config.doc_encoding) as r:
            doc = r.read()
            doc = [text for text in doc.split(self.config.retriever_sep) if text.strip()]
            if not self.config.replace_retriever_sep:
                doc = [self.config.retriever_sep + text for text in doc]

        retriever_doc_native = doc
        ranker_doc_native = []
        retriever_doc_translated = []
        ranker_doc_translated = []

        for text in doc:
            ranker_doc_native.extend(self._create_ranker_doc(text))

        if self.translator:
            retriever_doc_translated = [
                self.translator._translate(text) for text in tqdm(doc, desc='Translating docs paragraphs')
            ]
            ranker_doc_translated = [
                self.translator._translate(text) for text in tqdm(
                    ranker_doc_native, desc='Grouping and translating docs by paragraphs'
                )
            ]
  
        return {
            'docs': doc, 'retriever_docs_native': retriever_doc_native,
            'ranker_docs_native': ranker_doc_native, 'retriever_docs_translated': retriever_doc_translated,
            'ranker_docs_translated': ranker_doc_translated
        }

    def add_documents(self, docs_links: list):
        docs = []
        self.retriever_docs_native = []
        self.retriever_docs_translated = []
        self.ranker_docs_native = []
        self.ranker_docs_translated = []

        with open(f'{self.storage_path}/storage_data.json', 'r') as r:
            files = json.load(r)

        for link in docs_links:
            doc_name = os.path.split(link)[1].replace('.docx', '').replace('.rtf', '').replace('.doc', '')
            if doc_name not in files['documents']:
                new_path = self.doc_converter.convert_doc(link, f'{self.storage_path}/docs/{os.path.split(link)[1]}')
                files['documents'][doc_name] = self._process_doc(new_path)

        files['retriever_range'] = {}
        files['ranker_range'] = {}
        retriever_sum = 0
        ranker_sum = 0
        for doc_name in files['documents']:
            files['retriever_range'][retriever_sum+len(files['documents'][doc_name]['retriever_docs_native'])] = doc_name
            retriever_sum += len(files['documents'][doc_name]['retriever_docs_native'])
            files['ranker_range'][ranker_sum+len(files['documents'][doc_name]['ranker_docs_native'])] = doc_name
            ranker_sum += len(files['documents'][doc_name]['ranker_docs_native'])
        
        with open(f'{self.storage_path}/storage_data.json', 'w') as w:
            json.dump(files, w)

        for name in files['documents']:
            docs.extend(files['documents'][name]['docs'])
            self.retriever_docs_native.extend(files['documents'][name]['retriever_docs_native'])
            self.retriever_docs_translated.extend(files['documents'][name]['retriever_docs_translated'])
            self.ranker_docs_native.extend(files['documents'][name]['ranker_docs_native'])
            self.ranker_docs_translated.extend(files['documents'][name]['ranker_docs_translated'])
        
        self.retriever_range = files['retriever_range']
        self.ranker_range = files['ranker_range']

    def get_documents_names(self):
        with open(f'{self.storage_path}/storage_data.json', 'r') as r:
            json_decoded = json.load(r)

        return list(json_decoded['documents'].keys())

    def del_document(self, doc_name: str):
        docs = []
        self.retriever_docs_native = []
        self.retriever_docs_translated = []
        self.ranker_docs_native = []
        self.ranker_docs_translated = []

        with open(f'{self.storage_path}/storage_data.json', 'r') as r:
            files = json.load(r)

        assert doc_name in files['documents'], 'Document is not found in this storage'
        del files['documents'][doc_name]

        files['retriever_range'] = {}
        files['ranker_range'] = {}
        retriever_sum = 0
        ranker_sum = 0
        for doc_name in files['documents']:
            files['retriever_range'][retriever_sum+len(files['documents'][doc_name]['retriever_docs_native'])] = doc_name
            retriever_sum += len(files['documents'][doc_name]['retriever_docs_native'])
            files['ranker_range'][ranker_sum+len(files['documents'][doc_name]['ranker_docs_native'])] = doc_name
            ranker_sum += len(files['documents'][doc_name]['ranker_docs_native'])
        
        with open(f'{self.storage_path}/storage_data.json', 'w') as w:
            json.dump(files, w)

        for name in files['documents']:
            docs.extend(files['documents'][name]['docs'])
            self.retriever_docs_native.extend(files['documents'][name]['retriever_docs_native'])
            self.retriever_docs_translated.extend(files['documents'][name]['retriever_docs_translated'])
            self.ranker_docs_native.extend(files['documents'][name]['ranker_docs_native'])
            self.ranker_docs_translated.extend(files['documents'][name]['ranker_docs_translated'])
        
        self.retriever_range = files['retriever_range']
        self.ranker_range = files['ranker_range']

    def add_dataset(self, link: str, name: str, only_for_benchmark: bool = False):
        dataset = pd.read_csv(link)

        with open(f'{self.storage_path}/storage_data.json', 'r') as r:
            json_decoded = json.load(r)

        assert (name not in json_decoded['datasets']['train_data'] and
                name not in json_decoded['datasets']['benchmark_data']), 'Dataset name already exists'

        code = 'benchmark_data' if only_for_benchmark else 'train_data'

        json_decoded['datasets'][code][name] = {
            'translated_question': list(dataset['translated_question']),
            'translated_context': list(dataset['translated_context']),
            'native_context': list(dataset['native_context']),
            'native_question': list(dataset['native_question'])
        }

        with open(f'{self.storage_path}/storage_data.json', 'w') as w:
            json.dump(json_decoded, w)

    def del_dataset(self, name: str):
        with open(f'{self.storage_path}/storage_data.json') as r:
            json_decoded = json.load(r)

        assert (name in json_decoded['datasets']['train_data'] or
                name in json_decoded['datasets']['benchmark_data']), 'Dataset is not found'

        if name in json_decoded['datasets']['train_data']:
            del json_decoded['datasets']['train_data'][name]
        else:
            del json_decoded['datasets']['benchmark_data'][name]
        
        with open(f'{self.storage_path}/storage_data.json', 'w') as w:
            json.dump(json_decoded, w)
    
    def get_datasets_names(self):
        with open(f'{self.storage_path}/storage_data.json', 'r') as r:
            json_decoded = json.load(r)

        return {
            'train_data': list(json_decoded['datasets']['train_data'].keys()),
            'benchmark_data': list(json_decoded['datasets']['benchmark_data'].keys())
        }

    def make_data_loaders(
            self,
            val_size=0.2,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            seed=42
    ):
        with open(f'{self.storage_path}/storage_data.json', 'r') as r:
            json_decoded = json.load(r)

        train_data = self._unpack_data(json_decoded['datasets']['train_data'])

        dataset = BaseDataset(train_data)
        train_length = int(len(dataset) * (1 - val_size))
        val_length = len(dataset) - train_length

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_length, val_length], generator=torch.Generator().manual_seed(seed)
        )

        benchmark_data = self._unpack_data(json_decoded['datasets']['benchmark_data'])

        self.train_loader = list(DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        ))
        self.val_loader = list(DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers
        ))
        self.test_loader = list(DataLoader(
            benchmark_data, batch_size=1, num_workers=num_workers
        ))

    def _unpack_data(self, data):
        unpacked_data = []

        for name in data:
            unpacked_data.extend([
                {
                    'question': data[name]['translated_question'][i],
                    'context': data[name]['translated_context'][i],
                    'native_context': data[name]['native_context'][i],
                    'native_question': data[name]['native_question'][i],
                }
                for i in range(len(data[name]['native_context']))
            ])
        return unpacked_data

    def _create_ranker_doc(self, doc):
        if self.config.ranker_sep:
            return [text for text in doc.split(self.config.ranker_sep) if text]
        else:
            return [doc]
