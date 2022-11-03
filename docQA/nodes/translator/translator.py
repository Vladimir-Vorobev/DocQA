from docQA.utils.torch.dataset import BaseDataset
from docQA.utils.utils import seed_worker

import torch
from transformers import (
    FSMTForConditionalGeneration,
    FSMTTokenizer
)


class Translator:
    def __init__(self, model_name, device='cuda'):
        self.model = FSMTForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.config.max_length = 512
        self.model.config.num_beams = 5
        self.tokenizer = FSMTTokenizer.from_pretrained(model_name)
        self.device = device
        self.batch_size = 8

    def _translate(self, text: str):
        translated_text = ''

        sentences = [phragment for phragment in text.split('.') if phragment]

        translator_dataset = BaseDataset(sentences)
        translator_loader = torch.utils.data.DataLoader(
            translator_dataset, batch_size=self.batch_size, shuffle=False,
            generator=torch.Generator().manual_seed(42), worker_init_fn=seed_worker
        )

        for batch in translator_loader:
            # batch = [i for i in batch if i]
            input_ids = self.tokenizer.prepare_seq2seq_batch(
                batch, return_tensors='pt', max_length=512, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**input_ids)

            translated_text += '. '.join([translated for translated in self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )])

            del input_ids

        return translated_text
