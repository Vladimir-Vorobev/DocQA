from docQA.nodes.translator import translator

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


class QuestionGenerator:
    def __init__(
            self,
            cdqa_pipe=None,
            qa_model='bert-large-cased-whole-word-masking-finetuned-squad',
            qg_model='valhalla/t5-base-qg-hl',

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
            cuda=False,
    ):
        self.device = 'cuda' if cuda else 'cpu'

        self._cdqa_pipe = cdqa_pipe
        self._native_lang = native_lang
        self._qg_engine = qg_engine
        self._kw_model = KeyBERT()

        self._docs = []
        self._native_docs = []

        for doc, native_doc in zip(cdqa_pipe._retriever_docs_en, cdqa_pipe._retriever_docs):
            for text, native_text in zip(doc, native_doc):
                self._docs.append({'input_text': text, **qg_model_args})
                self._native_docs.append(native_text)

        if qg_model_path:
            raise BaseException('Sorry, but loading qg_model from your path is not supported now.')
        elif qg_engine == 'transformers':
            self._qg_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
            self._qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model).to(self.device)
        else:
            raise BaseException(f'Invalid qg_engine {qg_engine}. Supported types: transformers.')

        if qa_model_path:
            self._qa_model = joblib.load(qa_model_path)
            for arg in qa_model_args:
                setattr(self._qa_model.args, arg, qa_model_args[arg])
        elif qa_engine == 'transformers':
            self._qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)
            self._qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model).to(self.device)
        else:
            raise BaseException(f'Invalid qa_engine {qa_engine}. Supported types: transformers.')

    def __call__(self, use_ranker_retriever=True, use_qa=True, rr_threshold=0.5, save_questions=True,
                 path_to_save=None):
        if not use_ranker_retriever and not use_qa:
            raise BaseException('To generate_questions at least one of use_ranker_retriever or use_qa must be True.')

        questions = self._generate_questions(use_ranker_retriever, use_qa, rr_threshold, save_questions)

        if save_questions:
            pd.DataFrame(questions).to_csv(path_to_save)

    def _generate_questions(self, use_ranker_retriever, use_qa, retriever_threshold, save_questions):
        outputs = []

        for doc, native_doc in zip(tqdm(self._docs, ascii=True, desc='Generating questions'), self._native_docs):
            doc_text = doc['input_text']
            ##
            if len(doc_text.split()) < 7:
                continue
            ##

            keywords = self._extract_keywords(doc_text, (3, 5), 5)

            candidates = self._prepare_candidates(keywords, doc_text)

            questions = self.generate(candidates)

            if not questions:
                continue

            for question in questions:
                question_statement = question['question_statement']
                question_context = question['context']
                result = self._question_checker(question_statement, question_context, doc_text, native_doc,
                                                use_ranker_retriever, use_qa, retriever_threshold)

                if result:
                    outputs.append(result)

        return outputs

    def _extract_keywords(self, text, keyphrase_ngram_range, top_n):
        return self._kw_model.extract_keywords(text, keyphrase_ngram_range=keyphrase_ngram_range, use_maxsum=False,
                                               use_mmr=False, top_n=top_n)

    def _prepare_candidates(self, keywords, text):
        candidates = []

        for keyword in keywords:
            newText = text.replace(keyword[0], '<hl> ' + keyword[0] + ' <hl>') + '</s>'

            candidates.append(newText)

            newText = '<hl> ' + text.replace(keyword[0], '<hl> ' + keyword[0]) + '</s>'

            candidates.append(newText)

            newText = text.replace(keyword[0], keyword[0] + ' <hl>') + '<hl> </s>'

            candidates.append(newText)

        return candidates

    def generate(self, candidates):
        questions = []

        for candidate in candidates:

            tokens = self._qg_tokenizer.encode(candidate, return_tensors='pt', max_length=512, truncation=True).to(
                self.device)

            newgend = []
            gend = self._qg_model.generate(tokens, max_length=512)

            for gen in gend:
                newgend.append(self._qg_tokenizer.decode(gen, max_length=512, truncation=True))

            questions.append({'question_statement': newgend[0], 'context': candidate})

            del tokens
            torch.cuda.empty_cache()

        return questions

    def _question_checker(self, question_statement, question_answer, doc_text, native_doc, use_ranker_retriever, use_qa,
                          retriever_threshold):
        if use_ranker_retriever:
            for result in self._ranker_retriever_question_checker(question_statement, doc_text, native_doc,
                                                                  retriever_threshold):
                if use_qa:
                    result = self._qa_question_checker(doc_text, question_statement, question_answer, result=result)

                if result:
                    return result

        if use_qa and not use_ranker_retriever:
            result = self._qa_question_checker(doc_text, question_statement, question_answer)

            if result:
                return result

        return None

    def _ranker_retriever_question_checker(self, question_statement, doc_text, native_doc, retriever_threshold):
        rr_preds = self._cdqa_pipe(question_statement, return_en=True)
        native_question_statement = translator.translate(question_statement, from_code='en', to_code=self._native_lang)

        for rank, pred in enumerate(rr_preds):
            pred_answer = pred['answer']
            retriever_score = pred['retriever_score']
            ranker_score = pred['ranker_score']

            if (doc_text in pred_answer and retriever_score >= retriever_threshold):
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

        native_question_statement = translator.translate(question_statement, from_code='en', to_code=self._native_lang)
        native_question_answer = translator.translate(question_answer, from_code='en', to_code=self._native_lang)
        native_doc = translator.translate(doc_text, from_code='en', to_code=self._native_lang)

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
                    'native_question_answer': native_question_answer,
                }

            result['en_question_answer'] = question_answer
            result['native_question_answer'] = native_question_answer
            return result

        return None
