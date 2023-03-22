# DocQA
### Open source library for the CDQA task 
##### DocQA library provides simple tools for Retriever, Ranker, Translator, CatBoost and QG pipelines creation and usage. It is also possible to combine different pipelines in a general Pipeline to achieve your purposes in developing your QA-system.

To augment the dataset, we support the ability to generate questions for each paragraph of the text, as well as paraphrasing questions. Especially for this project, we made a dataset and a model based on our ChatGPT paraphrases.

Dataset link: [kaggle](https://www.kaggle.com/datasets/vladimirvorobevv/chatgpt-paraphrases) [hf](https://huggingface.co/datasets/humarin/chatgpt-paraphrases)
Model link: [hf](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base)

### Code examples:
**General pipeline usage**
```python
from docQA.pipelines import Pipeline
pipe = Pipeline(storage)
pipe.add_node(TranslatorPipeline, name='translator', is_technical=True, demo_only=True, num_beams=15)
pipe.add_node(RetrieverPipeline, name='retriever')
pipe.add_node(RankerPipeline, name='ranker')
pipe.add_node(CatboostPipeline, name='catboost')
input_text = 'Что такое персональные данные?'
pipe(input_text)
```
**Output:**
```python
[{'input': 'Что такое персональные данные?',
  'output': {'answers': [{'answer': '1) персональные данные - любая информация, относящаяся к прямо или косвенно определенному или определяемому физическому лицу (субъекту персональных данных);',
     'total_score': 0.7820469439029694,
     'scores': {'retriever_cos_sim': 0.7021254301071167,
      'ranker_cos_sim': 0.861968457698822}},
    {'answer': '3) предполагаемые пользователи персональных данных;',
     'total_score': 0.7196908891201019,
     'scores': {'retriever_cos_sim': 0.6950922012329102,
      'ranker_cos_sim': 0.7442895770072937}},
    {'answer': '2) цель обработки персональных данных;',
     'total_score': 0.6678484380245209,
     'scores': {'retriever_cos_sim': 0.6220505237579346,
      'ranker_cos_sim': 0.7136463522911072}},
    {'answer': '2) правовые основания и цели обработки персональных данных;',
     'total_score': 0.6541507244110107,
     'scores': {'retriever_cos_sim': 0.6059879660606384,
      'ranker_cos_sim': 0.7023134827613831}},
    {'answer': '4) цель обработки персональных данных;',
     'total_score': 0.6533105671405792,
     'scores': {'retriever_cos_sim': 0.6052820682525635,
      'ranker_cos_sim': 0.701339066028595}},
    {'answer': '7. Субъект персональных данных имеет право на получение информации, касающейся обработки его персональных данных, в том числе содержащей:',
     'total_score': 0.6530922055244446,
     'scores': {'retriever_cos_sim': 0.5903106331825256,
      'ranker_cos_sim': 0.7158737778663635}},
    {'answer': '2. Субъект персональных данных имеет право на защиту своих прав и законных интересов, в том числе на возмещение убытков и (или) компенсацию морального вреда в судебном порядке.',
     'total_score': 0.643451452255249,
     'scores': {'retriever_cos_sim': 0.6000968813896179,
      'ranker_cos_sim': 0.6868060231208801}},
    {'answer': '3) категории персональных данных;',
     'total_score': 0.6415583193302155,
     'scores': {'retriever_cos_sim': 0.5838664770126343,
      'ranker_cos_sim': 0.6992501616477966}},
    {'answer': '2) цель обработки персональных данных и ее правовое основание;',
     'total_score': 0.6184261739253998,
     'scores': {'retriever_cos_sim': 0.560569167137146,
      'ranker_cos_sim': 0.6762831807136536}},
    {'answer': '1.1) персональные данные, разрешенные субъектом персональных данных для распространения, - персональные данные, доступ неограниченного круга лиц к которым предоставлен субъектом персональных данных путем дачи согласия на обработку персональных данных, разрешенных субъектом персональных данных для распространения в порядке, предусмотренном настоящим Федеральным законом; (в ред. Федерального закона от 30.12.2020 N 519-ФЗ)',
     'total_score': 0.6150197982788086,
     'scores': {'retriever_cos_sim': 0.5444573163986206,
      'ranker_cos_sim': 0.6855822801589966}}]},
  'modified_input': 'What is Personal Data?'}]
```

### Statistics:


