from docQA.typing_schemas import PipeOutputElement

from typing import Union, List


def top_n_qa_error(contexts: List[str], outputs: List[PipeOutputElement], n: Union[int, List[int]]):
    if isinstance(n, int):
        n = [n]

    top_n_errors = {n_error: 0 for n_error in n}

    for context, output in zip(contexts, outputs):
        answers = [answer['answer'] for answer in output['output']['answers']]

        for top_n in n:
            top_n_errors[top_n] += 0 if context in answers[:top_n] else 1

    for top_n in n:
        top_n_errors[top_n] /= len(contexts)

    return top_n_errors
