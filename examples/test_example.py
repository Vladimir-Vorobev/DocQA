from docQA.pipelines import Pipeline, TranslatorPipeline, RetrieverPipeline, RankerPipeline
import pandas as pd

pipe = Pipeline(['docs/Федеральный-закон-от-27.07.2006-N-152-ФЗ-О-персональных-данных.txt'])
pipe.add_node(TranslatorPipeline, name='translator', is_technical=True, model_name='facebook/wmt19-ru-en')
pipe.add_node(RetrieverPipeline, name='retriever')
pipe.add_node(RankerPipeline, name='ranker')

df = pd.read_csv('docs/generated_questions_2.1.csv')
train_df = [{'question': df.iloc[i]['en_question'], 'context': df.iloc[i]['en_context']} for i in range(len(df))]
pipe.fit(train_df)
