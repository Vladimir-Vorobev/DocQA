from docQA.pipelines import pipeline

import pandas as pd
import catboost


documents = [
    r'C:\Users\Вова\Downloads\Telegram Desktop\Федеральный_закон_от_27_07_2006_N_152_ФЗ_О_персональных_данных_—.txt',
]

pipe = pipeline(documents)

cat_clf = catboost.CatBoostClassifier()
cat_clf.load_model('catboost')

while True:
    question = input('Enter your question... ')
    translated_question = pipe._preprocess(question)[0]
    result = pipe(question)
    for i in range(len(result)):
        result[i]['catboost_score'] = cat_clf.predict_proba(pd.Series({
            'question': translated_question,
            'answer': result[i]['answer'],
            'ranker_score': result[i]['ranker_score'],
            'retriever_score': result[i]['retriever_score'],
            'index': i + 1,
        }))[1]

    print(sorted(result, key=lambda x: x['ranker_score'] + x['retriever_score'] + x['catboost_score'], reverse=True))
