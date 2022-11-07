from docQA.utils.torch.dataset import BaseDataset
from docQA.utils.utils import seed_worker
from docQA import seed

import torch
from transformers import (
    FSMTForConditionalGeneration,
    FSMTTokenizer
)


class Translator:
    def __init__(self, model_name, max_length=512, num_beams=5, batch_size=8, device='cuda'):
        self.model = FSMTForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.config.max_length = max_length
        self.model.config.num_beams = num_beams
        self.tokenizer = FSMTTokenizer.from_pretrained(model_name)
        self.device = device
        self.batch_size = batch_size

    def _translate(self, text: str):
        translated_text = ''

        sentences = [fragment for fragment in text.split('.') if fragment]

        translator_dataset = BaseDataset(sentences)
        translator_loader = torch.utils.data.DataLoader(
            translator_dataset, batch_size=self.batch_size, shuffle=False,
            generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker
        )

        for batch in translator_loader:
            input_ids = self.tokenizer(
                batch, return_tensors='pt', max_length=self.model.config.max_length , padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**input_ids)

            translated_text += '. '.join([translated for translated in self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )])

            del input_ids

        return translated_text
