from docQA.errors import QgError
from docQA.pipelines import TranslatorPipeline

from keybert import KeyBERT
from transformers import (
    AutoModel,
    AutoTokenizer,
    T5TokenizerFast,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)
import torch
import pandas as pd
from tqdm.autonotebook import tqdm
from typing import Any


class QuestionGenerator:
    """
    A question generator for an artificial dataset generation
    """
    def __init__(
            self,
            pipe: Any = None,
            qa_model: str = 'bert-large-cased-whole-word-masking-finetuned-squad',
            qg_model: str = 'valhalla/t5-base-qg-hl',
            kw_model: str = 'sentence-transformers/gtr-t5-large',
            back_translator_model: str = 'facebook/wmt19-en-ru',

            qg_model_args: dict = {
                'max_questions': 5,
            },

            keyword_range: tuple = (3, 5),
            keyword_qty: int = 8,
            min_tokens_qty: int = 7,
            min_rr_threshold: float = 0.3,
            max_rr_threshold: float = 0.91,
            device: str = 'cuda',
    ):
        """
        :param pipe: a Pipeline class
        :param qa_model: a question answering model name/path to be used as one of the generated question validators
        :param qg_model: a question generation model name/path to generate questions (not recommended to change)
        :param kw_model: a keyword extractor model name/path to extract keywords from text to be used as answers for qg_model
        :param back_translator_model: a translator model name/path to be used as back translator if translation is used
        in Pipeline. If not - set this parameter to None
        :param qg_model_args: a dict of args for qg_model
        :param keyword_range: a range of keyword phrase length in ngrams to be used as an answer in qg_model
        :param keyword_qty: a maximum number of keywords to extract from each retriever text from a storage
        :param min_tokens_qty: a minimum length of text to be used to extract keywords
        :param min_rr_threshold: a minimum cosine similarity score for generated question to be accepted by Retriever-Ranker algorithm
        :param max_rr_threshold: a maximum cosine similarity score for generated question to be accepted by Retriever-Ranker algorithm
        :param device: a device to be used by models
        """
        self.device = device

        self.pipe = pipe
        self.kw_model = KeyBERT(kw_model)
        self.back_translator = TranslatorPipeline(back_translator_model) if back_translator_model else None

        self.keyword_range = keyword_range
        self.keyword_qty = keyword_qty
        self.min_tokens_qty = min_tokens_qty
        self.min_rr_threshold = min_rr_threshold
        self.max_rr_threshold = max_rr_threshold

        self.docs = []
        self.native_docs = []

        for text, native_text in zip(pipe.storage.retriever_docs_translated, pipe.storage.retriever_docs_native):
            self.docs.append({'input_text': text, **qg_model_args})
            self.native_docs.append(native_text)

        self.qg_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model).to(self.device)

        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model).to(self.device)

    def generate_questions(
            self,
            path_to_save: str = 'dataset.csv',
            use_ranker_retriever: bool = False,
            use_qa: bool = True
    ):
        """
        A method to generate artificial dataset for documents from a pipe storage
        :param path_to_save: a path where to save a generated dataset
        :param use_ranker_retriever: a flag if user wants to use Retriever-Ranker algorithm to validate a generated question or not
        :param use_qa: a flag if user wants to use QA-model to validate generated question or not
        """
        assert use_ranker_retriever or use_qa, QgError(
            'To generate_questions at least one of use_ranker_retriever or use_qa must be True.'
        )

        outputs = []

        for doc, native_doc in zip(tqdm(self.docs, ascii=True, desc='Generating questions'), self.native_docs):
            doc_text = doc['input_text']
            if len(doc_text.split()) < self.min_tokens_qty:
                continue

            keywords = self.extract_keywords(doc_text, self.keyword_range, self.keyword_qty)

            candidates = self.prepare_candidates(keywords, doc_text)

            if not candidates:
                continue

            questions = self.generate(candidates)

            if not questions:
                continue

            for question in questions:
                question_statement = question['question_statement'].replace('<unk>', '').replace('hl>', '').replace('  ', ' ').strip()
                question_context = question['context']
                result = self.question_checker(
                    question_statement, question_context, doc_text, native_doc,
                    use_ranker_retriever, use_qa, self.min_rr_threshold, self.max_rr_threshold
                )

                if result:
                    outputs.append(result)
        
        pd.DataFrame(outputs).to_csv(path_to_save)

    def extract_keywords(self, text: str, key_phrase_ngram_range: tuple, top_n: int):
        """
        A method to extract keywords from a text to use them as answers for QG-model
        :param text: a text from which keywords are extracted
        :param key_phrase_ngram_range: extract phrases with this ngram range
        :param top_n: a maximum number of keywords to extract
        :return: a list of extracted keyword phrases
        """
        return self.kw_model.extract_keywords(
            text, keyphrase_ngram_range=key_phrase_ngram_range, use_maxsum=False, use_mmr=False, top_n=top_n
        )

    @staticmethod
    def prepare_candidates(keywords: list, text: str):
        """
        A technical method to prepare keywords to be used in QG-model
        :param keywords: a list of generated keywords
        :param text: a text from which keywords have been generated
        :return: a list of candidates which can be used in QG-model
        """
        candidates = []

        for keyword in keywords:
            new_text = text.replace(keyword[0], '<hl> ' + keyword[0] + ' <hl>') + '</s>'

            candidates.append(new_text)

            new_text = '<hl> ' + text.replace(keyword[0], '<hl> ' + keyword[0]) + '</s>'

            candidates.append(new_text)

            new_text = text.replace(keyword[0], keyword[0] + ' <hl>') + '<hl> </s>'

            candidates.append(new_text)

        return candidates

    def generate(self, candidates: list):
        """
        A method where QG-model generates questions from candidates
        :param candidates: a list of extracted and preprocessed candidates
        :return: a list of dicts with generated question statements for contexts
        """
        questions = []

        tokens = self.qg_tokenizer(
            candidates, return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to('cuda')

        with torch.no_grad():
            outputs = self.qg_model.generate(**tokens, max_new_tokens=512)
        
        for gen in range(len(outputs)-1):
            decoded = self.qg_tokenizer.decode(outputs[gen], max_length=512, truncation=True).replace('<pad> ', '').replace('</s>', '').replace('<pad>', '').replace('<pad>', '')
            if decoded not in candidates[gen].replace('<hl> ', '').replace(' <hl>', '').replace('</s>', ''):
                questions.append((decoded, candidates[gen].replace('<hl> ', '').replace(' <hl>', '').replace('</s>', '')))

        del tokens
        torch.cuda.empty_cache()

        return [{'question_statement': question[0], 'context': question[1]} for question in set(questions)]

    def question_checker(
            self,
            question_statement: str,
            question_answer: str,
            doc_text: str,
            native_doc: str,
            use_ranker_retriever: bool,
            use_qa: bool,
            min_rr_threshold: float,
            max_rr_threshold: float
    ):
        """
        An algorithm to check if generated question is acceptable or not
        :param question_statement: a generated question
        :param question_answer: a generated answer for the question statement by QG-model
        :param doc_text: a context for which the question has been generated
        :param native_doc: a native version of doc_text
        :param use_ranker_retriever: a flag to use Retriever-Ranker algorithm for question validation or not
        :param use_qa: a flag to use QA-model for question validation or not
        :param min_rr_threshold: minimum cosine similarity score for generated question to be accepted by Retriever-Ranker algorithm
        :param max_rr_threshold: a maximum cosine similarity score for generated question to be accepted by Retriever-Ranker algorithm
        :return: a dict item to write in a dataset
        """
        if use_ranker_retriever:
            for result in self.ranker_retriever_question_checker(question_statement, doc_text, native_doc, min_rr_threshold, max_rr_threshold):
                if use_qa:
                    result = self.qa_question_checker(
                        question_statement, doc_text, native_doc, question_answer, result=result
                    )

                if result:
                    return result

        if use_qa and not use_ranker_retriever:
            result = self.qa_question_checker(question_statement, doc_text, native_doc, question_answer)

            if result:
                return result

    def ranker_retriever_question_checker(
            self,
            question_statement: str,
            doc_text: str,
            native_doc: str,
            min_rr_threshold: float,
            max_rr_threshold: float
    ):
        """
        A Retriever-Ranker algorithm to validate a generated question
        :param question_statement: a generated question
        :param doc_text: a context for which the question has been generated
        :param native_doc: a native version of doc_text
        :param min_rr_threshold: minimum cosine similarity score for generated question to be accepted by Retriever-Ranker algorithm
        :param max_rr_threshold: a maximum cosine similarity score for generated question to be accepted by Retriever-Ranker algorithm
        :return: a dict item to write in a dataset
        """
        rr_preds = self.pipe(question_statement, return_translated=True)
        native_question_statement = self.back_translator(question_statement, standardized=False)[0] if \
            self.back_translator else question_statement

        for rank, pred in enumerate(rr_preds[0]['output']['answers']):
            pred_answer = pred['translated_answer']
            retriever_score = pred['scores']['retriever_cos_sim']
            ranker_score = pred['scores']['ranker_cos_sim']
            score = (retriever_score + ranker_score) / 2

            if (doc_text in pred_answer and score >= min_rr_threshold and score <= max_rr_threshold and question_statement not in doc_text):
                yield {
                    'translated_question': question_statement,
                    'native_question': native_question_statement,
                    'translated_context': doc_text,
                    'native_context': native_doc,
                    'retriever_score': retriever_score,
                    'ranker_score': ranker_score,
                    'rank': rank + 1,
                }

    def qa_question_checker(
            self,
            question_statement: str,
            doc_text: str,
            native_doc: str,
            question_answer: str,
            result: dict = None
    ):
        """
        A QA-model algorithm to validate a generated question
        :param question_statement: a generated question
        :param doc_text: a context for which the question has been generated
        :param native_doc: a native version of doc_text
        :param question_answer: a generated answer for the question statement by QG-model
        :param result: a previous validator result or None
        :return: a dict item to write in a dataset or None
        """
        inputs = self.qa_tokenizer(question_statement, doc_text, return_tensors="pt").to('cuda')

        try:
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
        except:
            return

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        qa_pred_answer = self.qa_tokenizer.decode(predict_answer_tokens).lower()
        native_question_statement = self.back_translator(question_statement, standardized=False)[0]
        native_question_answer = self.back_translator(question_answer, standardized=False)[0]

        del inputs
        torch.cuda.empty_cache()

        if qa_pred_answer in question_answer or question_answer in qa_pred_answer:
            if not result:
                return {
                    'translated_question': question_statement,
                    'native_question': native_question_statement,
                    'translated_context': doc_text,
                    'native_context': native_doc,
                    'translated_question_answer': question_answer,
                    'native_question_answer': native_question_answer
                }

            result['translated_question_answer'] = question_answer
            result['native_question_answer'] = native_question_answer
            return result
