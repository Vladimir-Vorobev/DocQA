from docQA.typing_schemas import PipeOutput
from docQA.pipelines.base import BasePipeline
from docQA.errors import PipelineError
from docQA.metrics import top_n_qa_error
from docQA import seed

from tqdm.autonotebook import tqdm
from typing import List, Union
import json
import catboost
import numpy as np
import pandas as pd


class CatboostPipeline(BasePipeline):
    pipe_type = 'catboost'

    def __init__(
            self,
            texts: List[str],
            native_texts: List[str],
            weight: float = 1.0,
            return_num: int = 10,
            config_path: str = 'docQA/configs/catboost_config.json',
            name: str = 'catboost'
    ):
        BasePipeline.__init__(self)
        self.name = name
        self.texts = texts
        self.native_texts = native_texts
        self.weight = weight
        self.return_num = return_num
        self.model = catboost.CatBoostClassifier(iterations=250, random_state=seed)

    def __call__(
            self,
            data: PipeOutput,
            return_num: int = 10
    ) -> PipeOutput:
        if self.return_num != return_num and return_num == 30:
            return_num = self.return_num

        data = self.standardize_input(data)
        data = self.add_standard_answers(data, len(self.texts))

        for index in range(len(data)):
            answers = data[index]['output']['answers']
            question = data[index]['modified_input']

            for answer_index in range(len(answers)):
                answer = answers[answer_index]
                input_data = {
                    'question': question,
                    'answer': self.texts[answer['index']],
                    **answer['scores'],
                    'index': answer_index
                }

                score = self.model.predict_proba(pd.DataFrame(input_data, index=[0]))[0][1] * self.weight

                answer['scores'][f'{self.name}_proba'] = score
                answer['total_score'] += score
                answer['weights_sum'] += self.weight

            data[index]['output']['answers'] = \
                sorted(answers, key=lambda x: x['total_score'], reverse=True)[:return_num]

        return data

    def fit(
            self,
            data,
            train_dataset,
            val_dataset,
            top_n_errors=None,
            pipe=None
    ):
        # assert previous_outputs and previous_outputs[0]['output']['answers'] \
        #        and previous_outputs[0]['output']['answers'][0]['scores'], PipelineError(
        #            'To use CatboostPipeline you should have non-technical nodes in Pipeline before this pipeline.'
        #        )

        if not top_n_errors or not pipe:
            top_n_errors = {}

        data = {item['question']: item['native_context'] for item in data}

        train_dataset = self.standardize_input(train_dataset)
        train_dataset = self.add_standard_answers(train_dataset, len(self.texts))
        val_dataset = self.standardize_input(val_dataset)
        val_dataset = self.add_standard_answers(val_dataset, len(self.texts))

        train_top_n_errors = {}
        val_top_n_errors = {}

        for train_item in train_dataset:
            for answer in train_item['output']['answers']:
                answer['is_correct'] = 1 if self.native_texts[answer['index']] == data[train_item['input']] else 0

        for val_item in val_dataset:
            for answer in val_item['output']['answers']:
                answer['is_correct'] = 1 if self.native_texts[answer['index']] == data[val_item['input']] else 0

        train_dataset, val_dataset = self._create_dataset(train_dataset), self._create_dataset(val_dataset)

        X_train, y_train = train_dataset.drop('is_correct', axis=1), train_dataset['is_correct']
        X_val, y_val = val_dataset.drop('is_correct', axis=1), train_dataset['is_correct']

        for _ in tqdm(range(1), desc=f'Fine tuning {self.pipe_type}'):
            self.model.fit(X_train, y_train, text_features=['question', 'answer'], silent=True)

            train_questions = np.unique(X_train['question'])
            native_contexts = [data[question] for question in train_questions]
            pred_contexts = [pipe.__call__(question, threshold=0, is_demo=False)[0] for question in train_questions]
            train_top_n_errors = top_n_qa_error(native_contexts, pred_contexts, top_n_errors)

            if pipe:
                val_questions = np.unique(X_val['question'])
                native_contexts = [data[question] for question in val_questions]
                pred_contexts = [pipe.__call__(question, threshold=0, is_demo=False)[0] for question in val_questions]
                val_top_n_errors = top_n_qa_error(native_contexts, pred_contexts, top_n_errors)

        with open(f'docs/{self.name}_fitting_results.json', 'w') as w:
            w.write(json.dumps({
                'train_top_n_errors_history': train_top_n_errors,
                'val_top_n_errors_history': val_top_n_errors,
            }))

        return train_top_n_errors, val_top_n_errors

    def _create_dataset(self, df):
        dataset = []

        for item in df:
            question = item['modified_input']
            answers = item['output']['answers']
            for index, answer in enumerate(answers):
                answer_index = answer['index']
                translated_answer = self.texts[answer_index]
                scores = answer['scores']
                is_correct = answer['is_correct']

                new_train_item = {
                    'question': question,
                    'answer': translated_answer,
                    **scores,
                    'index': index,
                    'is_correct': is_correct
                }

                dataset.append(new_train_item)

        return pd.DataFrame(dataset)
