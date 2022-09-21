from docQA.utils.torch import BaseDataset
from docQA.pipelines import pipeline
from docQA.utils import seed_worker
from docQA import seed

from sklearn.model_selection import GridSearchCV
import torch
import catboost_pipe
import pandas as pd

documents = [
    r'C:\Users\Вова\Downloads\Telegram Desktop\Федеральный_закон_от_27_07_2006_N_152_ФЗ_О_персональных_данных_—.txt',
]

pipe = pipeline(documents)

df = pd.read_csv('../nodes/models/question_generator/generated_questions_2.1.csv')
docs = [i[0] if i else '' for i in pipe._retriever_docs]
train_df = [{'question': df.iloc[i]['en_question'], 'context': pipe._retriever_docs_en[docs.index(df.iloc[i]['native_context'])][0]} for i in range(len(df))]
dataset = BaseDataset(train_df)
train_length = int(len(dataset) * 0.7)
val_length = len(dataset) - train_length
train_df, val_df = torch.utils.data.random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(42))
train_dataset, val_dataset = [], []

for batch in train_df:
    question = batch['question']
    context = batch['context']
    for index, result in enumerate(pipe(question, return_en=True), start=1):
        answer = result['answer']
        ranker_score = result['ranker_score']
        retriever_score = result['retriever_score']

        is_correct = 1 if answer == context else 0
        train_dataset.append({
            'question': question,
            'answer': answer,
            'ranker_score': ranker_score,
            'retriever_score': retriever_score,
            'index': index,
            'is_correct': is_correct,
        })

for batch in val_df:
    question = batch['question']
    context = batch['context']
    for index, result in enumerate(pipe(question, return_en=True), start=1):
        answer = result['answer']
        ranker_score = result['ranker_score']
        retriever_score = result['retriever_score']

        is_correct = 1 if answer == context else 0
        val_dataset.append({
            'question': question,
            'answer': answer,
            'ranker_score': ranker_score,
            'retriever_score': retriever_score,
            'index': index,
            'is_correct': is_correct,
        })

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=42, shuffle=True, generator=torch.Generator().manual_seed(42), worker_init_fn=seed_worker)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=42, shuffle=True, generator=torch.Generator().manual_seed(42), worker_init_fn=seed_worker)

X_train, X_test, y_train, y_test = [], [], [], []

for batch in train_loader:
    for question, answer, ranker_score, retriever_score, index, is_correct in zip(batch['question'], batch['answer'], batch['ranker_score'], batch['retriever_score'], batch['index'], batch['is_correct']):
        X_train.append({'question': question, 'answer': answer, 'ranker_score': ranker_score.item(), 'retriever_score': retriever_score.item(), 'index': index.item()})
        y_train.append(is_correct.item())

X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)

for batch in val_loader:
    for question, answer, ranker_score, retriever_score, index, is_correct in zip(batch['question'], batch['answer'], batch['ranker_score'], batch['retriever_score'], batch['index'], batch['is_correct']):
        X_test.append({'question': question, 'answer': answer, 'ranker_score': ranker_score.item(), 'retriever_score': retriever_score.item(), 'index': index.item()})
        y_test.append(is_correct.item())

X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

cat_clf = catboost.CatBoostClassifier(random_state=seed)

parameters = {
    'iterations': [100],
    'max_depth': [1, 3, 5, 7, 10],
    'subsample': [0.75],
}

gs_cat_clf = GridSearchCV(cat_clf,
                  parameters,
                  scoring='recall',
                  cv=2)

gs_cat_clf.fit(X_train, y_train, text_features=['question', 'answer'])

cat_clf = catboost.CatBoostClassifier(random_state=seed, **gs_cat_clf.best_params_)
cat_clf.fit(X_train, y_train, text_features=['question', 'answer'])
cat_clf.save_model('catboost')
