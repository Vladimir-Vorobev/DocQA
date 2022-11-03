import torch
from transformers import (
    FSMTForConditionalGeneration,
    FSMTTokenizer
)
from docQA.utils.torch.dataset import BaseDataset
from docQA.utils.utils import seed_worker


class Translator:
    def __init__(self, model_name, device='cuda'):
        self.model = FSMTForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.config.max_length = 512
        self.model.config.num_beams = 5
        self.tokenizer = FSMTTokenizer.from_pretrained(model_name)
        self.device = device
        self.batch_size = 8

        if not hasattr(self, 'translate_text'):
            self.translate_text = True

    def _translate(self, doc):
        if not self.translate_text:
            return doc

        translated_doc = []

        paragraphs = doc.split('\n')
        for paragraph in paragraphs:
            sentences = [i for i in paragraph.split('.') if i]
            translator_dataset = BaseDataset(sentences)
            translator_loader = torch.utils.data.DataLoader(
                translator_dataset, batch_size=self.batch_size, shuffle=False,
                generator=torch.Generator().manual_seed(42), worker_init_fn=seed_worker
            )

            for batch in translator_loader:
                batch = [i for i in batch]
                input_ids = self.tokenizer.prepare_seq2seq_batch(
                    batch, return_tensors="pt", max_length=512, truncation=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(**input_ids)

                translated_doc.append(
                    '. '.join([i for i in self.tokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )])
                )

                del input_ids

        return '\n'.join(translated_doc)
