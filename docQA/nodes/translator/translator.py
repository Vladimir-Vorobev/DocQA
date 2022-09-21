import torch
from transformers import (
    FSMTForConditionalGeneration,
    FSMTTokenizer
)


class Translator:
    def __init__(self, model_name, device='cuda'):
        self.model = FSMTForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = FSMTTokenizer.from_pretrained(model_name)
        self.device = device

    def _translate(self, doc):
        translated_doc = []

        for text in doc.split('\n'):
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids)
            translated_doc.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

            # разобраться к кэшированием GPU и поправить
            del input_ids
            torch.cuda.empty_cache()

        return '\n'.join(translated_doc)
