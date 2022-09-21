from docQA.pipelines import pipeline
from docQA.nodes.models import QuestionGenerator


documents = [
    r'C:\Users\Вова\Downloads\Telegram Desktop\Федеральный_закон_от_27_07_2006_N_152_ФЗ_О_персональных_данных_—.txt',
]

pipe = pipeline(documents)
qg_pipe = QuestionGenerator(cdqa_pipe=pipe, cuda=True)
qg_pipe(path_to_save='test.csv')
