class ModifyOutputMixin:
    @staticmethod
    def modify_output(data, texts, translated_texts=None, return_translated=False):
        if isinstance(data, dict):
            return data

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
