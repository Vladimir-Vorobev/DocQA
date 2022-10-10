from docQA.typing_schemas import PipeOutput
from docQA.utils.torch import BaseDataset
from docQA.pipelines.base import BasePipeline
from docQA.errors import PipelineError
from docQA import seed

from typing import List, Union
import torch
import catboost
import pandas as pd


class CatboostPipeline(BasePipeline):
    pipe_type = 'catboost'

    def __init__(
            self,
            texts: List[str],
            native_texts: List[str],
            weight: float = 1.0,
            number: int = 0,
            config_path: str = 'docQA/configs/catboost_config.json'
    ):
        BasePipeline.__init__(self)
        self.texts = texts
        self.native_texts = native_texts
        self.weight = weight
        self.number = number
        self.model = catboost.CatBoostClassifier(iterations=250, random_state=seed)

    def __call__(
            self,
            data: PipeOutput,
            catboost_n: int = 10
    ) -> PipeOutput:
        data = self.standardize_input(data)

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

                answer['scores'][f'catboost_{self.number}_proba'] = score
                answer['total_score'] += score
                answer['weights_sum'] += self.weight

            data[index]['output']['answers'] = \
                sorted(answers, key=lambda x: x['total_score'], reverse=True)[:catboost_n]

        return data

    def fit(
            self,
            data,
            previous_outputs,
            val_size=0.3,
    ):
        assert previous_outputs and previous_outputs[0]['output']['answers'] \
               and previous_outputs[0]['output']['answers'][0]['scores'], PipelineError(
                   'To use CatboostPipeline you should have non-technical nodes in Pipeline before this pipeline.'
               )
        data = {item['question']: item['native_context'] for item in data}

        for output_item in previous_outputs:
            for answer in output_item['output']['answers']:
                answer['is_correct'] = 1 if self.native_texts[answer['index']] == data[output_item['input']] else 0

        dataset = BaseDataset(previous_outputs)
        train_length = int(len(dataset) * (1 - val_size))
        val_length = len(dataset) - train_length
        train_df, val_df = torch.utils.data.random_split(
            dataset, [train_length, val_length], generator=torch.Generator().manual_seed(seed)
        )
        train_dataset, val_dataset = self._create_dataset(train_df), self._create_dataset(val_df)

        X_train, y_train = train_dataset.drop('is_correct', axis=1), train_dataset['is_correct']
        X_test, y_test = val_dataset.drop('is_correct', axis=1), val_dataset['is_correct']

        self.model.fit(X_train, y_train, text_features=['question', 'answer'], silent=True)

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
