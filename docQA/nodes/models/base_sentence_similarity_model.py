from docQA.configs import ConfigParser
from docQA.utils.torch import BaseDataset
from docQA.utils.visualization import fine_tune_plot
from docQA.utils import seed_worker
from docQA.errors import DeviceError, SentenceEmbeddingsModelError

import joblib
import torch
import json

import sentence_transformers.util as util
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
)

from tqdm.autonotebook import tqdm, trange
import numpy as np


class BaseSentenceSimilarityEmbeddingsModel:
    def __init__(
            self,
            optimizer=None,
            loss_func=None,
            config_path='',
            name='',
    ):
        self.name = name
        self._config = ConfigParser(config_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.best_model = None
        self.optimizer = optimizer
        self.loss_func = loss_func

        if self.config.model_path:
            self.model = joblib.load(self.config.model_path)
        else:
            self.model = AutoModel.from_pretrained(self.config.model_name).to(self.config.device)

        if not self.optimizer:
            self.optimizer = AdamW(self.model.parameters(), self.config.lr)
        if not self.loss_func:
            self.loss_func = torch.nn.CrossEntropyLoss()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        if config['is_training']:
            self.model.train()

            if not self.optimizer:
                self.optimizer = AdamW(self.model.parameters(), self.config.lr)
            if not self.loss_func:
                self.loss_func = torch.nn.CrossEntropyLoss()
        else:
            self.model.eval()

        # удалить при поддержке cpu
        if 'device' in config:
            assert config['device'] != 'cuda', DeviceError(config['device'])

        for arg in config:
            setattr(self._config, arg, config[arg])

        self.model = self.model.to(self.config.device)

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu()

    def fine_tune(self, train_data, val_size=0.2, pipe=None):
        assert self.config.is_training, SentenceEmbeddingsModelError('training')
        self.model.train()

        dataset = BaseDataset(train_data)
        train_length = int(len(dataset) * (1 - val_size))
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_length, val_length],
                                                                   generator=torch.Generator().manual_seed(42))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training_batch_size,
                                                   shuffle=True, generator=torch.Generator().manual_seed(42),
                                                   worker_init_fn=seed_worker)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.training_batch_size, shuffle=True,
                                                 generator=torch.Generator().manual_seed(42),
                                                 worker_init_fn=seed_worker)

        train_loss_history = []
        val_loss_history = []
        train_top_1_error_history = []
        val_top_1_error_history = []
        train_top_3_error_history = []
        val_top_3_error_history = []
        val_top_1_error_max = 10

        for epoch in tqdm(range(self.config.epochs), desc=f'Fine tuning {self.name}',
                          display=self.config.display_progress_bar):
            train_loss_sum, train_top_1_error, train_top_3_error = self.epoch_step(train_loader, pipe=pipe)

            with torch.no_grad():
                val_loss_sum, val_top_1_error, val_top_3_error = self.epoch_step(val_loader, train=False, pipe=pipe)

            train_loss_sum /= self.config.training_batch_size
            val_loss_sum /= self.config.training_batch_size
            train_top_1_error /= train_length
            val_top_1_error /= val_length
            train_top_3_error /= train_length
            val_top_3_error /= val_length

            train_loss_history.append(train_loss_sum)
            val_loss_history.append(val_loss_sum)
            train_top_1_error_history.append(train_top_1_error)
            val_top_1_error_history.append(val_top_1_error)
            train_top_3_error_history.append(train_top_3_error)
            val_top_3_error_history.append(val_top_3_error)
            fine_tune_plot({'train loss': train_loss_history, 'val loss': val_loss_history}, self.name,
                                 x_label='epoch', y_label='loss')
            fine_tune_plot(
                {'train top-1 error': train_top_1_error_history, 'val top-1 error': val_top_1_error_history}, self.name,
                metric='top-1 error', x_label='epoch', y_label='top-1 error')
            fine_tune_plot(
                {'train top-3 error': train_top_3_error_history, 'val top-3 error': val_top_3_error_history}, self.name,
                metric='top-3 error', x_label='epoch', y_label='top-3 error')

            if val_top_1_error <= val_top_1_error_max:
                self.best_model = self.model
                val_top_1_error_max = val_top_1_error

            with open(f'/content/drive/MyDrive/Sber DPO/{self.name}_fine_tune_results.json', 'w') as w:
                w.write(json.dumps({
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'train_top_1_error_history': train_top_1_error_history,
                    'val_top_1_error_history': val_top_1_error_history,
                    'train_top_3_error_history': train_top_3_error_history,
                    'val_top_3_error_history': val_top_3_error_history,
                }))

        self.model, self.best_model = self.best_model, None
        return train_loss_history, val_loss_history

    def epoch_step(self, loader, train=True, pipe=None):
        questions = []
        contexts = []
        loss_sum = 0
        top_1_error = 0
        top_3_error = 0
        for batch in loader:
            questions.extend(batch['question'])
            contexts.extend(batch['context'])

            question = self.tokenizer([item for item in batch['question']], return_tensors="pt",
                                      max_length=self.config.max_length, truncation=True, padding="max_length")
            context = self.tokenizer([item for item in batch['context']], return_tensors="pt",
                                     max_length=self.config.max_length, truncation=True, padding="max_length")

            embeddings_question = self.forward(**question.to(self.config.device))
            embeddings_context = self.forward(**context.to(self.config.device))
            scores = torch.mm(embeddings_question, torch.transpose(embeddings_context, 0, 1)) * self.config.cossim_scale
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_question.device)
            loss = (self.loss_func(scores, labels) + self.loss_func(torch.transpose(scores, 0, 1), labels)) / 2

            if train:
                self.optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                self.optimizer.step()

            loss_sum += float(loss)

        if pipe:
            self.config.is_training = False
            self.model.eval()

            for question, context in zip(questions, contexts):
                pred_contexts = pipe.__call__(question, return_en=True)
                if pred_contexts[0]['answer'] != context:
                    top_1_error += 1
                if context not in [pred['answer'] for pred in pred_contexts[:3]]:
                    top_3_error += 1

            self.config.is_training = True
            self.model.train()

        return loss_sum, top_1_error, top_3_error

    def encode(self, sentences):
        assert not self.config.is_training, SentenceEmbeddingsModelError('evaluating')
        self.model.eval()

        all_embeddings = []
        length_sorted_idx = np.argsort([self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), self.config.evaluation_batch_size, desc="Batches", disable=True):
            sentences_batch = sentences_sorted[start_index:start_index + self.config.evaluation_batch_size]
            features = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt')
            features = util.batch_to_device(features, self.config.device)

            with torch.no_grad():
                out_features = self.forward(**features)
                all_embeddings.extend(out_features)

        return [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def _text_length(text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):
            return 1
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            return sum([len(t) for t in text])
