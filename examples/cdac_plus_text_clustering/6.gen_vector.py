#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import random
import sys
from typing import List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

format = '[%(asctime)s] %(levelname)s \t [%(filename)s %(lineno)d] %(message)s'
logging.basicConfig(format=format,
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from project_settings import project_path
from toolbox.torch.utils.data.dataset.text_classifier_json_dataset import TextClassifierJsonDataset
from toolbox.torch.utils.data.vocabulary import Vocabulary
from toolbox.torch.utils.data.tokenizers.pretrained_bert_tokenizer import PretrainedBertTokenizer
from toolbox.torchtext.models.text_clustering.cdac_plus import BertForConstrainClustering, CDACPlus
from toolbox.torchtext.models.text_clustering.cdac_plus import StudentsTDistribution, AuxiliaryTargetDistribution
from toolbox.torchtext.models.text_clustering.utils import clustering_score


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_all', default='train_all.json', type=str)
    parser.add_argument('--vocabulary', default='vocabulary', type=str)

    parser.add_argument('--all_vector', default='all_vector.json', type=str)
    parser.add_argument('--n_clusters', default=200, type=int)

    parser.add_argument('--pretrained_model_dir', default='chinese-bert-wwm-ext', type=str)
    parser.add_argument('--pretrain_model_filename', default='./finetune/best.bin', type=str)

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
            label: str = sample['label']

            tokens: List[int] = [self.vocab.get_token_index(token, namespace=self.tokens_namespace) for token in tokens]
            tokens = self.vocab.pad_or_truncate_ids_by_max_length(tokens, max_length=max_token_length)
            label: int = self.vocab.get_token_index(label, namespace=self.labels_namespace)

            batch_tokens.append(tokens)
            batch_labels.append(label)

        batch_tokens = torch.from_numpy(np.array(batch_tokens))
        batch_labels = torch.from_numpy(np.array(batch_labels))

        return batch_tokens, batch_labels


def main():
    args = get_args()

    model_name = args.pretrained_model_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocabulary = Vocabulary.from_files(args.vocabulary)

    # dataset
    train_all_dataset = TextClassifierJsonDataset(
        json_file=args.train_all,
        tokenizer=PretrainedBertTokenizer(model_name),
    )
    collate_fn = CollateFunction(
        vocab=vocabulary,
        token_min_padding_length=5,
    )

    # model
    bert_for_constrain_clustering = BertForConstrainClustering.from_pretrained(model_name)

    model = CDACPlus(
        backbone=bert_for_constrain_clustering,
        hidden_size=768,
        dropout=0.1,
        n_clusters=args.n_clusters,
    )

    with open(args.pretrain_model_filename, 'rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    count = 0
    with open(args.all_vector, 'w', encoding='utf-8') as f:
        for instance in train_all_dataset:
            input_ids, targets = collate_fn([instance])
            input_ids = input_ids.to(device)
            with torch.no_grad():
                logits = model.forward(input_ids)

            logits = logits.detach().cpu().numpy()

            row = {
                'text': instance['metadata']['text'],
                'tokens': instance['tokens'],
                'label': instance['label'],
                'vector': logits.tolist()
            }
            row = json.dumps(row, ensure_ascii=False)
            f.write('{}\n'.format(row))

            if count % 1000 == 0:
                logger.info('count: {}'.format(count))

            count += 1

    return


if __name__ == '__main__':
    main()
