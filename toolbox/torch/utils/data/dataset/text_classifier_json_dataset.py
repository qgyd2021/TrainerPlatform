#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import os
from typing import Callable, List

from torch.utils.data import Dataset

from toolbox.torch.utils.data.vocabulary import Vocabulary
from toolbox.torch.utils.data.tokenizers.tokenizer import Tokenizer


class TextClassifierJsonDataset(Dataset):
    def __init__(self,
                 json_file: str,
                 tokenizer: Tokenizer,
                 ):
        self.json_file = json_file
        self.tokenizer = tokenizer

        samples = list()
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)

                text = row['text']
                label = row.get('label')
                samples.append({
                    'text': text,
                    'label': label,
                })

        self.samples = samples

    def __getitem__(self, index):
        sample = self.samples[index]
        text = sample['text']
        label = sample['label']

        instance = self.text_to_instance(text, label)

        return instance

    def __len__(self):
        return len(self.samples)

    def text_to_instance(self, text: str, label: str = None):

        tokens: List[str] = self.tokenizer.tokenize(text)

        result = {'tokens': tokens, 'metadata': {'text': text}}
        if label is not None:
            result['label'] = label

        return result


class HierarchicalClassificationJsonDataset(Dataset):
    def __init__(self,
                 json_file: str,
                 tokenizer: Tokenizer,
                 n_hierarchical: int = 2,
                 max_sequence_length: int = None,
                 ):
        self.json_file = json_file
        self.tokenizer = tokenizer

        self.n_hierarchical = n_hierarchical
        self.max_sequence_length = max_sequence_length

        samples = list()
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)

                text = row['text']
                labels = [row.get("label{}".format(idx), None) for idx in range(self.n_hierarchical)]
                if all(labels):
                    label = '_'.join(labels)
                else:
                    label = None
                samples.append({
                    'text': text,
                    'label': label,
                })
        self.samples = samples

    def __getitem__(self, index):
        sample = self.samples[index]
        text = sample['text']
        label = sample['label']

        instance = self.text_to_instance(text, label)

        return instance

    def __len__(self):
        return len(self.samples)

    def _truncate(self, tokens):
        if len(tokens) > self.max_sequence_length:
            tokens = tokens[:self.max_sequence_length]
        return tokens

    def text_to_instance(self, text: str, label: str = None):

        tokens: List[str] = self.tokenizer.tokenize(text)
        if self.max_sequence_length is not None:
            tokens = self._truncate(tokens)

        result = {'tokens': tokens, 'metadata': {'text': text}}
        if label is not None:
            result['labels'] = label

        return result


if __name__ == '__main__':
    pass
