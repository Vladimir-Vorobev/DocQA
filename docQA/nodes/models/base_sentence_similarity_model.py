from docQA.mixins import ModifyOutputMixin
from docQA.configs import ConfigParser
from docQA.utils.visualization import visualize_fitting
from docQA.utils import seed_worker, batch_to_device
from docQA.errors import DeviceError, SentenceEmbeddingsModelError
from docQA.metrics import top_n_qa_error
from docQA import seed

import joblib
import torch
import json

from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
)

from tqdm.autonotebook import tqdm, trange
import numpy as np
from copy import deepcopy


class BaseSentenceSimilarityEmbeddingsModel(ModifyOutputMixin):
    def __init__(
            self,
            model=None,
            optimizer=None,
            loss_func=None,
            config_path='',
            name='',
            child_state=None,
    ):
        self.child_state = child_state
        self.name = name
        self.config_path = config_path
        self._config = ConfigParser(config_path)

        if model:
            self.config.model_name = model

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        self.optimizer = optimizer
        self.loss_func = loss_func

        self.autocast_type = torch.float32

        if self.config.model_path:
            self.model = joblib.load(self.config.model_path)
        else:
            self.model = AutoModel.from_pretrained(
                self.config.model_name, local_files_only=self.config.local_files_only
            ).to(self.config.device)

        if not self.optimizer:
            self.optimizer = AdamW(self.model.parameters(), self.config.lr)
        if not self.loss_func:
            self.loss_func = torch.nn.CrossEntropyLoss()

        self.best_model = self.model

    def __getstate__(self):
        return {
            'model': self.config.model_name,
            'optimizer': self.optimizer,
            'loss_func': self.loss_func,
            'config_path': self.config_path,
            'name': self.name,
            'child_state': self.child_state
        }

    def __setstate__(self, state):
        if type(self) == type(BaseSentenceSimilarityEmbeddingsModel):
            self.__init__(**state)

        else:
            child_state = state['child_state']

            BaseSentenceSimilarityEmbeddingsModel.__init__(self, **state)
            self.texts = child_state['texts']
            self.weight = child_state['weight']
            self.return_num = child_state['return_num']

            if 'retriever_pipeline.RetrieverPipeline' in str(type(self)):
                self.embeddings = self.encode(child_state['texts'])

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

    def fit(
            self,
            train_dataset,
            val_dataset,
            train_previous_outputs,
            val_previous_outputs,
            native_texts,
            translated_texts,
            top_n_errors=None,
            node=None,
            eval_step=5
    ):
        if not top_n_errors or not node:
            top_n_errors = []

        self.config.is_training = True
        self.model.train()

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.training_batch_size,
            shuffle=True, generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.training_batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker
        )

        train_loss_history = []
        val_loss_history = []
        train_top_n_errors_history = {n: [] for n in top_n_errors}
        val_top_n_errors_history = {n: [] for n in top_n_errors}

        for epoch in tqdm(range(self.config.epochs), desc=f'Fine tuning {self.name}'):
            do_evaluation = epoch == 0 or not (epoch + 1) % eval_step

            train_loss_sum, train_top_n_errors = self.epoch_step(
                train_loader, train_previous_outputs, native_texts, translated_texts,
                top_n_errors=top_n_errors, node=node
            )

            with torch.no_grad():
                val_loss_sum, val_top_n_errors = self.epoch_step(
                    val_loader, val_previous_outputs, native_texts, translated_texts,
                    top_n_errors=top_n_errors, node=node, train=False
                )

            train_loss_sum /= self.config.training_batch_size
            val_loss_sum /= self.config.training_batch_size

            train_loss_history.append(train_loss_sum)
            val_loss_history.append(val_loss_sum)

            visualize_fitting(
                {
                    'train loss': train_loss_history,
                    'val loss': val_loss_history
                }, self.name, x_label='epoch', y_label='loss'
            )

            if do_evaluation:
                for n in top_n_errors:
                    train_top_n_errors_history[n].append(train_top_n_errors[n])
                    val_top_n_errors_history[n].append(val_top_n_errors[n])

                    visualize_fitting(
                        {
                            f'train top-{n} error': train_top_n_errors_history[n],
                            f'val top-{n} error': val_top_n_errors_history[n]
                        }, self.name, metric=f'top-{n} error', x_label='epoch', y_label=f'top-{n} error'
                    )

            # if val_top_1_error <= val_top_1_error_max:
            #     self.best_model = self.model
            #     val_top_1_error_max = val_top_1_error

            with open(f'docs/{self.name}_fitting_results.json', 'w') as w:
                w.write(json.dumps({
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'train_top_n_errors_history': train_top_n_errors_history,
                    'val_top_n_errors_history': val_top_n_errors_history,
                }))

        # self.model, self.best_model = self.best_model, None
        self.config.is_training = False
        self.model.eval()

        return train_loss_history, val_loss_history

    def epoch_step(self, loader, previous_outputs, native_texts, translated_texts, top_n_errors, node, train=True):
        validation_data = {}
        epoch_top_n_errors = {}
        loss_sum = 0

        for batch in loader:
            for question, native_context in zip(batch['question'], batch['native_context']):
                validation_data[question] = native_context

            question = self.tokenizer(
                [item for item in batch['question']], return_tensors="pt",
                max_length=self.config.max_question_length, truncation=True, padding="max_length"
            )

            context = self.tokenizer(
                [item for item in batch['context']], return_tensors="pt",
                max_length=self.config.max_length, truncation=True, padding="max_length"
            )

            with torch.autocast(device_type="cuda", dtype=self.autocast_type):
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

        if node:
            self.config.is_training = False
            self.model.eval()

            return_num = max(top_n_errors)

            with torch.autocast(device_type="cuda", dtype=self.autocast_type):
                questions = [output['modified_input'] for output in previous_outputs]
                native_contexts = [validation_data[question] for question in questions]
                pred_contexts = node(deepcopy(previous_outputs), return_num=return_num)
                epoch_top_n_errors = top_n_qa_error(
                    native_contexts,
                    self.modify_output(pred_contexts, native_texts, translated_texts), top_n_errors
                )

            self.config.is_training = True
            self.model.train()

        return loss_sum, epoch_top_n_errors

    def encode(self, sentences):
        assert not self.config.is_training, SentenceEmbeddingsModelError('evaluating')
        self.model.eval()

        all_embeddings = []
        length_sorted_idx = np.argsort([self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), self.config.evaluation_batch_size, desc="Batches", disable=True):
            sentences_batch = sentences_sorted[start_index:start_index + self.config.evaluation_batch_size]
            features = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt',max_length=self.config.max_length)
            features = batch_to_device(features, self.config.device)

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
        if isinstance(text, dict):
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):
            return 1
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            return sum([len(t) for t in text])
