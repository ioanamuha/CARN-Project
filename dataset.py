import json
import re

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class AmbiStoryDataset(Dataset):
    def __init__(self, json_file, model_name='bert-base-uncased', max_len=256):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    # def __init__(self, json_file, word_to_ix, max_len=250):
    #     with open(json_file, 'r') as f:
    #         self.data = json.load(f)
    #
    #     self.keys = list(self.data.keys())
    #     self.word_to_ix = word_to_ix
    #     self.max_len = max_len

    def __len__(self):
        return len(self.keys)

    def vectorize(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
        indices = [self.word_to_ix.get(t, self.word_to_ix["<UNK>"]) for t in tokens]
        return indices

    def pad_sequence(self, indices):
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        sample_id = self.keys[idx]
        entry = self.data[sample_id]

        # story_text = f"{entry.get('precontext', '')} {entry['sentence']} {entry.get('ending', '')}"

        # marked_sentence = re.sub(
        #     rf"\b({re.escape(homonym)})\b",
        #     r"<H> \1 </H>",
        #     sentence,
        #     flags=re.IGNORECASE
        # )

        homonym = entry['homonym']
        sentence = entry['sentence']

        def_text = f"{homonym} : {entry['judged_meaning']}"
        story_text = f"{entry.get('precontext', '')} {sentence} {entry.get('ending', '')}"

        story_enc = self.tokenizer(story_text, return_tensors='pt', padding='max_length', truncation=True,
                                   max_length=self.max_len)
        def_enc = self.tokenizer(def_text, return_tensors='pt', padding='max_length', truncation=True, max_length=32)

        inputs = self.tokenizer(
            def_text,
            story_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        return {
            'id': sample_id,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'story_ids': story_enc['input_ids'].squeeze(0),
            'story_mask': story_enc['attention_mask'].squeeze(0),
            'def_ids': def_enc['input_ids'].squeeze(0),
            'def_mask': def_enc['attention_mask'].squeeze(0),
            'target': torch.tensor(float(entry['average']), dtype=torch.float)
        }

        # story_indices = self.vectorize(story_text)
        # def_indices = self.vectorize(def_text)
        #
        # story_len = max(1, min(len(story_indices), self.max_len))
        # def_len = max(1, min(len(def_indices), self.max_len))
        #
        # return {
        #     'id': sample_id,
        #     'story_input': self.pad_sequence(story_indices),
        #     'story_len': torch.tensor(story_len, dtype=torch.long),
        #     'def_input': self.pad_sequence(def_indices),
        #     'def_len': torch.tensor(def_len, dtype=torch.long),
        #     'target': torch.tensor(float(entry['average']), dtype=torch.float)
        # }
