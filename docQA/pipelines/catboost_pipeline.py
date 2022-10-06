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
            weight: float = 1.0,
            number: int = 0,
            config_path: str = 'docQA/configs/catboost_config.json'
    ):
        BasePipeline.__init__(self)
        self.weight = weight
        self.number = number
        self.model = catboost.CatBoostClassifier(random_state=seed)

    def __call__(
            self,
            data: PipeOutput,
            catboost_n: int = 10
    ) -> PipeOutput:
        data = self.standardize_input(data)

        for index in range(len(data)):
            answers = data[index]['output']['answers']

            for answer_index in range(len(answers)):
                answer = answers[answer_index]
                input_data = answer['scores']
                input_data['answer'] = answer['answer']
                input_data['index'] = answer_index + 1

                score = self.model.predict_proba(input_data)[1] * self.weight

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

        for output_item, data_item in zip(previous_outputs, data):
            output_item['context'] = data_item['native_context']

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

    @staticmethod
    def _create_dataset(df):
        dataset = []

        for item in df:
            question = item['modified_input']
            context = item['context']
            answers = item['output']['answers']
            for index, result in enumerate(answers, start=1):
                answer = result['answer']
                translated_answer = result['translated_answer'] if 'translated_answer' in result else answer
                scores = result['scores']
                is_correct = 1 if answer == context else 0

                new_train_item = {
                    'question': question,
                    'answer': translated_answer,
                    'index': index,
                    'is_correct': is_correct
                }

                for score in scores:
                    new_train_item[score] = scores[score]

                dataset.append(new_train_item)

        return pd.DataFrame(dataset)
