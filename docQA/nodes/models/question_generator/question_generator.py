from docQA.nodes.translator import Translator

from keybert import KeyBERT
from transformers import (
    AutoModel,
    AutoTokenizer,
    T5TokenizerFast,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)
import torch
import joblib
import pandas as pd
from tqdm.autonotebook import tqdm

translator = Translator('facebook/wmt19-ru-en')


class QuestionGenerator:
    def __init__(
            self,
            cdqa_pipe=None,
            qa_model='bert-large-cased-whole-word-masking-finetuned-squad',
            qg_model='../t5-base-qg-hl_',

            qg_model_path=None,
            qa_model_path=None,

            qg_model_args={
                'max_questions': 5,
            },

            qa_model_args={
                'max_seq_length': 512,
                'silent': True,
            },

            qg_engine='transformers',
            qa_engine='transformers',

            native_lang='ru',
            device='cuda',
    ):
        self.device = device

        self._cdqa_pipe = cdqa_pipe
        self._native_lang = native_lang
        self._qg_engine = qg_engine
        self._kw_model = KeyBERT('../all-MiniLM-L6-v2_')

        self.keyword_range = (3, 5)
        self.keyword_qty = 8
        self.min_tokens_qty = 7
        self.min_rr_treshold = 0.3
        self.max_rr_treshold = 0.91

        self._docs = []
        self._native_docs = []

        for text, native_text in zip(cdqa_pipe.preprocessor.retriever_docs_translated, cdqa_pipe.preprocessor.retriever_docs_native):
            self._docs.append({'input_text': text, **qg_model_args})
            self._native_docs.append(native_text)

        if qg_model_path:
            raise BaseException('Sorry, but loading qg_model from your path is not supported now.')
        elif qg_engine == 'transformers':
            self._qg_tokenizer = T5TokenizerFast.from_pretrained('../t5-base')
            self._qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model).to(self.device)
        else:
            raise BaseException(f'Invalid qg_engine {qg_engine}. Supported types: transformers.')

        # if qa_model_path:
        #     self._qa_model = joblib.load(qa_model_path)
        #     for arg in qa_model_args:
        #         setattr(self._qa_model.args, arg, qa_model_args[arg])
        # elif qa_engine == 'transformers':
        #     self._qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)
        #     self._qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model).to(self.device)
        # else:
        #     raise BaseException(f'Invalid qa_engine {qa_engine}. Supported types: transformers.')

    def _generate_questions(self, use_ranker_retriever=True, use_qa=False, save_questions=True, path_to_save=None):
        if not use_ranker_retriever and not use_qa:
            raise BaseException('To generate_questions at least one of use_ranker_retriever or use_qa must be True.')

        outputs = []

        for doc, native_doc in zip(tqdm(self._docs, ascii=True, desc='Generating questions'), self._native_docs):
            doc_text = doc['input_text']
            ##
            if len(doc_text.split()) < self.min_tokens_qty:
                continue
            ##

            keywords = self._extract_keywords(doc_text, self.keyword_range, self.keyword_qty)

            candidates = self._prepare_candidates(keywords, doc_text)

            questions = self.generate(candidates)

            if not questions:
                continue

            for question in questions:
                question_statement = question['question_statement']
                question_context = question['context']
                result = self._question_checker(
                    question_statement, question_context, doc_text, native_doc,
                    use_ranker_retriever, use_qa, self.min_rr_treshold, self.max_rr_treshold
                )

                if result:
                    outputs.append(result)
        
        if save_questions:
            pd.DataFrame(outputs).to_csv(path_to_save)

    def _extract_keywords(self, text, keyphrase_ngram_range, top_n):
        return self._kw_model.extract_keywords(
            text, keyphrase_ngram_range=keyphrase_ngram_range, use_maxsum=False, use_mmr=False, top_n=top_n
        )

    def _prepare_candidates(self, keywords, text):
        candidates = []

        for keyword in keywords:
            new_text = text.replace(keyword[0], '<hl> ' + keyword[0] + ' <hl>') + '</s>'

            candidates.append(new_text)

            new_text = '<hl> ' + text.replace(keyword[0], '<hl> ' + keyword[0]) + '</s>'

            candidates.append(new_text)

            new_text = text.replace(keyword[0], keyword[0] + ' <hl>') + '<hl> </s>'

            candidates.append(new_text)

        return candidates

    def generate(self, candidates):
        questions = []

        tokens = self._qg_tokenizer.batch_encode_plus(
            candidates, return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to('cuda')

        with torch.no_grad():
            outputs = self._qg_model.generate(**tokens, max_new_tokens=512)
        
        for gen in range(len(outputs)-1):
            decoded = self._qg_tokenizer.decode(outputs[gen], max_length=512, truncation=True).replace('<pad> ', '').replace('</s>', '').replace('<pad>', '').replace('<pad>', '')
            if decoded not in candidates[gen].replace('<hl> ', '').replace(' <hl>', '').replace('</s>', ''):
                questions.append((decoded, candidates[gen].replace('<hl> ', '').replace(' <hl>', '').replace('</s>', '')))

        del tokens
        torch.cuda.empty_cache()

        return [{'question_statement': question[0], 'context': question[1]} for question in set(questions)]

    def _question_checker(
            self, question_statement, question_answer, doc_text, native_doc, use_ranker_retriever, use_qa, min_rr_threshold, max_rr_threshold
    ):
        if use_ranker_retriever:
            for result in self._ranker_retriever_question_checker(question_statement, doc_text, native_doc, min_rr_threshold, max_rr_threshold):
                if use_qa:
                    result = self._qa_question_checker(doc_text, question_statement, question_answer, result=result)

                if result:
                    return result

        if use_qa and not use_ranker_retriever:
            result = self._qa_question_checker(doc_text, question_statement, question_answer)

            if result:
                return result

        return None

    def _ranker_retriever_question_checker(self, question_statement, doc_text, native_doc, min_rr_treshold, max_rr_treshold):
        rr_preds = self._cdqa_pipe(question_statement, return_translated=True)
        native_question_statement = translator._translate(question_statement)

        for rank, pred in enumerate(rr_preds[0]['output']['answers']):
            pred_answer = pred['translated_answer']
            retriever_score = pred['scores']['retriever_cos_sim']
            ranker_score = pred['scores']['ranker_cos_sim']
            score = (retriever_score + ranker_score) / 2

            if (doc_text in pred_answer and score >= min_rr_treshold and score <= max_rr_treshold and question_statement not in doc_text):
                yield {
                    'en_question': question_statement,
                    'native_question': native_question_statement,
                    'en_context': doc_text,
                    'native_context': native_doc,
                    'retriever_score': retriever_score,
                    'ranker_score': ranker_score,
                    'rank': rank + 1,
                }

    def _qa_question_checker(self, doc_text, question_statement, question_answer, result=None):
        inputs = self._qa_tokenizer(question_statement, doc_text, return_tensors="pt").to('cuda')

        with torch.no_grad():

            outputs = self._qa_model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        qa_pred_answer = self._qa_tokenizer.decode(predict_answer_tokens).lower()
        native_question_statement = translator._translate(question_statement)
        native_question_answer = translator._translate(question_answer)
        native_doc = translator._translate(doc_text)

        del inputs
        torch.cuda.empty_cache()

        if qa_pred_answer in question_answer or question_answer in qa_pred_answer:
            if not result:
                return {
                    'en_question': question_statement,
                    'native_question': native_question_statement,
                    'en_context': doc_text,
                    'native_context': native_doc,
                    'en_question_answer': question_answer,
                    'native_question_answer': native_question_answer
                }

            result['en_question_answer'] = question_answer
            result['native_question_answer'] = native_question_answer
            return result

        return None
