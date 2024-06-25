#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

from project_settings import project_path
from toolbox.torch.utils.data.dataset.text_classifier_json_dataset import HierarchicalClassificationJsonDataset
from toolbox.torch.utils.data.tokenizers.pretrained_bert_tokenizer import PretrainedBertTokenizer
from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='model_dir', type=str)
    parser.add_argument('--pretrained_model_dir', default='pretrained_model_dir', type=str)
    parser.add_argument('--text', default='this is a test text', type=str)
    args = parser.parse_args()
    return args


class CollateFunction(object):
    def __init__(self,
                 vocab: Vocabulary,
                 token_min_padding_length: int = 0,
                 tokens_namespace: str = 'tokens',
                 labels_namespace: str = 'labels',
                 ):
        self.vocab = vocab
        self.token_min_padding_length = token_min_padding_length
        self.tokens_namespace = tokens_namespace
        self.labels_namespace = labels_namespace

    def __call__(self, batch: List[dict]):

        max_token_length = max([len(sample['tokens']) for sample in batch])
        if max_token_length < self.token_min_padding_length:
            max_token_length = self.token_min_padding_length

        batch_tokens = list()
        batch_labels = list()
        for sample in batch:
            tokens: List[str] = sample['tokens']
            labels: str = sample['labels']

            tokens: List[int] = [self.vocab.get_token_index(token, namespace=self.tokens_namespace) for token in tokens]
            tokens = self.vocab.pad_or_truncate_ids_by_max_length(tokens, max_length=max_token_length)
            labels: int = self.vocab.get_token_index(labels, namespace=self.labels_namespace)

            batch_tokens.append(tokens)
            batch_labels.append(labels)

        batch_tokens = torch.from_numpy(np.array(batch_tokens))
        batch_labels = torch.from_numpy(np.array(batch_labels))

        return batch_tokens, batch_labels


def test_jit_model(text: str, **kwargs):
    model = kwargs['model']
    vocabulary = kwargs['vocabulary']
    tokenizer = kwargs['tokenizer']

    index_to_token = vocabulary.get_index_to_token_vocabulary(namespace='labels')

    tokens: List[str] = tokenizer.tokenize(text)
    tokens: List[int] = [vocabulary.get_token_index(token, namespace='tokens') for token in tokens]
    batch_tokens = [tokens]
    batch_tokens = torch.from_numpy(np.array(batch_tokens))

    outputs = model.forward(batch_tokens)

    probs = outputs['probs']

    label_idx = torch.argmax(probs, dim=-1)
    label_idx = label_idx.numpy()
    prob = probs[0][label_idx].detach().numpy()

    label_str = index_to_token[label_idx[0]]

    result = {
        'label': label_str,
        'prob': round(float(prob), 4)
    }
    return result


def load_model(model_dir: str, pretrained_model_dir: str):
    model_dir = Path(model_dir)
    model_path = model_dir / 'final.zip'
    vocabulary_path = model_dir / 'vocabulary'

    model = torch.jit.load(model_path)
    vocabulary = Vocabulary.from_files(vocabulary_path)
    tokenizer = PretrainedBertTokenizer(pretrained_model_dir)

    result = {
        'model': model,
        'vocabulary': vocabulary,
        'tokenizer': tokenizer,
    }
    return result


def main():
    args = get_args()

    kwargs = load_model(args.model_dir, args.pretrained_model_dir)
    outputs = test_jit_model(args.text, **kwargs)
    print(outputs)

    return


if __name__ == '__main__':
    main()
