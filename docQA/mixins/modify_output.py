from docQA.typing_schemas import PipeOutputElement

from typing import List


class ModifyOutputMixin:
    """
    A mixin for BasePipeline and BaseSentenceSimilarityModel
    """
    @staticmethod
    def modify_output(
            data: List[PipeOutputElement],
            texts: list,
            translated_texts: list = None,
            return_translated: bool = False
    ):
        """
        Modifies pipelines outputs indexes into texts
        :param data: a list of PipeOutputElement elements
        :param texts: a list of native texts from storage
        :param translated_texts: translated versions of texts if user wants to see them instead of native versions
        :param return_translated: a flag if user wants to see translated texts
        :return:
        """
        for item in data:
            for answer_index in range(len(item['output']['answers'])):
                answer = item['output']['answers'][answer_index]
                answer['total_score'] /= answer['weights_sum'] if answer['weights_sum'] else 1
                new_answer = {'answer': texts[answer['index']]}

                if return_translated and translated_texts:
                    new_answer['translated_answer'] = translated_texts[answer['index']]

                del answer['index'], answer['weights_sum']
                new_answer.update(answer)
                item['output']['answers'][answer_index] = new_answer

        return data
