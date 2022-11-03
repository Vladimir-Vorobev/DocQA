import os
import sys

docqa_path = 'C:\\Users\\Вова\\PycharmProjects\\DocQA'
sys.path.append(docqa_path)
os.chdir(docqa_path)

from docQA.pipelines import Pipeline, TranslatorPipeline, RetrieverPipeline, RankerPipeline, CatboostPipeline
import pandas as pd

pipe = Pipeline()
pipe.add_node(TranslatorPipeline, name='translator', is_technical=True, demo_only=True, model_name='facebook/wmt19-ru-en')
pipe.add_node(RetrieverPipeline, name='retriever-1', model='sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-trained-scratch', return_num=100)
# pipe.add_node(RetrieverPipeline, name='retriever-2', model='sentence-transformers/multi-qa-mpnet-base-dot-v1', return_num=50)
# pipe.add_node(RankerPipeline, name='ranker-1', model='sentence-transformers/all-MiniLM-L6-v2', return_num=50)
# pipe.add_node(RankerPipeline, name='ranker-2', model='sentence-transformers/all-mpnet-base-v2', return_num=30)
# pipe.add_node(RankerPipeline, name='ranker-3', model='sentence-transformers/all-roberta-large-v1', return_num=10)
# pipe.add_node(CatboostPipeline, name='catboost')

# df = pd.read_csv('docs/generated_questions_2.1.csv')
# train_df = [{
#     'question': df.iloc[i]['en_question'],
#     'context': df.iloc[i]['en_context'],
#     'native_context': df.iloc[i]['native_context']
# } for i in range(len(df))]
#
# pipe.fit(train_df)
#
# while True:
#     print(pipe(input('Введите вопрос: ')))

# pipe.save()

import pickle
k = pickle.dumps(pipe)

p = pickle.loads(k)
# p.load(r'C:\Users\Вова\PycharmProjects\DocQA\pipeline.pkl')

print(p('Кто такой оператор?'))

# import os
# print(os.path.isfile(r'C:\Users\Вова/.cache\huggingface\transformers\31323203fc8c2ec251d539773484877f83a37b65524d540de5a2d946c5a1708c.56209d2ca3707ce9263f4035ac7a3a3903fdda4180df9f4174972e23e045b436'))
#
# from docQA.pipelines import Pipeline, TranslatorPipeline, RetrieverPipeline, RankerPipeline, CatboostPipeline, QgPipeline
# import pandas as pd
#
# pipe = Pipeline(['docs/152.txt', 'docs/115.txt', 'docs/262.txt'])
# pipe.add_node(TranslatorPipeline, name='translator', is_technical=True, demo_only=True, model_name='Helsinki-NLP/opus-mt-ru-en', translate_text=False)
# pipe.add_node(RetrieverPipeline, name='retriever')
# pipe.add_node(RankerPipeline, name='ranker')
#
# qg_pipe = Pipeline(['docs/152.txt', 'docs/115.txt', 'docs/262.txt'])
# qg_pipe.add_node(QgPipeline, cdqa_pipe=pipe, name='qg')
# qg_pipe(data='docs/generated_questions_3.csv', return_output=False)
