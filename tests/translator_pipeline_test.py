from docQA.pipelines import TranslatorPipeline

import pytest

pipe = TranslatorPipeline()


def test_translator_pipeline_1():
    input_data, output_data = 'Привет!', 'Hello!'
    assert pipe(input_data, standardized=False)[0] == output_data


def test_translator_pipeline_2():
    input_data, output_data = 'Привет!', 'Hello!'
    assert pipe(input_data) == [{'input': input_data, 'output': {'answers': []}, 'modified_input': output_data}]


def test_translator_pipeline_3():
    input_data = ['Привет!', 'Я работаю в Сбере', 'Где мой зачет за IVR?']
    output_data = ['Hello!', 'I work in Sber', 'Where is my IVR credit?']
    assert pipe(input_data) == [
        {'input': input_data_item, 'output': {'answers': []}, 'modified_input': output_data_item}
        for input_data_item, output_data_item in zip(input_data, output_data)
    ]


def test_translator_pipeline_4():
    assert pipe('') == [{'input': '', 'output': {'answers': []}, 'modified_input': ''}]
