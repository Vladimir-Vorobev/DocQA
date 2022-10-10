import numpy as np
import torch
import random

seed = 42

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from docQA.pipelines import Pipeline, TranslatorPipeline, RetrieverPipeline, RankerPipeline, CatboostPipeline
import pandas as pd


pipe = Pipeline(['docs/Федеральный-закон-от-27.07.2006-N-152-ФЗ-О-персональных-данных.txt'])
pipe.add_node(TranslatorPipeline, name='translator', is_technical=True, model_name='facebook/wmt19-ru-en')
pipe.add_node(RetrieverPipeline, name='retriever')
pipe.add_node(RankerPipeline, name='ranker')
pipe.add_node(CatboostPipeline, name='catboost')

df = pd.read_csv('docs/generated_questions_2.1.csv')
train_df = [{'question': df.iloc[i]['en_question'], 'context': df.iloc[i]['en_context'], 'native_context': df.iloc[i]['native_context']} for i in range(len(df))]
pipe.fit(train_df)

while True:
    print(pipe(input('Введите вопрос: ')))
