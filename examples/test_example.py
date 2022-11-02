from docQA.pipelines import Pipeline, TranslatorPipeline, RetrieverPipeline, RankerPipeline, CatboostPipeline
import pandas as pd


pipe = Pipeline(['docs/152.txt'])
pipe.add_node(TranslatorPipeline, name='translator', is_technical=True, demo_only=True, model_name='facebook/wmt19-ru-en')
pipe.add_node(RetrieverPipeline, name='retriever')
pipe.add_node(RankerPipeline, name='ranker')
pipe.add_node(CatboostPipeline, name='catboost')

df = pd.read_csv('docs/generated_questions_2.1.csv')
train_df = [{'question': df.iloc[i]['en_question'], 'context': df.iloc[i]['en_context'], 'native_context': df.iloc[i]['native_context']} for i in range(len(df))]
pipe.fit(train_df)
