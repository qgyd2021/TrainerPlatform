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

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from project_settings import project_path
from toolbox.torch.utils.data.vocabulary import Vocabulary
from toolbox.torch.utils.data.tokenizers.pretrained_bert_tokenizer import PretrainedBertTokenizer
from toolbox.torchtext.models.text_clustering.cdac_plus import BertForConstrainClustering, CDACPlus


def get_args():
    """
    python 7.faiss_test.py \
    --vocabulary ./file_folder_name/vocabulary \
    --all_vector ./file_folder_name/all_vector.json \
    --pretrain_model_filename ./file_folder_name/finetune/best.bin
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocabulary', default='vocabulary', type=str)

    parser.add_argument('--all_vector', default='all_vector.json', type=str)
    parser.add_argument('--n_clusters', default=200, type=int)

    parser.add_argument('--pretrained_model_dir', default='../../pretrained_models/chinese-bert-wwm-ext', type=str)
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


class FaissRetrieval(object):
    def __init__(self,
                 all_vector_json: str,
                 embedding_dim: int,
                 sim_mode: str = 'cosine',
                 top_k: int = 40
                 ):
        self.all_vector_json = all_vector_json
        self.embedding_dim = embedding_dim
        self.sim_mode = sim_mode
        self.top_k = top_k

        self.index = faiss.IndexFlatL2(embedding_dim)

        self.vector_list: np.ndarray = None
        self.text_info_list: List[dict] = None

        self._init_index()

    def _init_index(self):
        logger.info('[init index] start')

        vector_list = list()
        text_info_list = list()

        count = 0
        with open(self.all_vector_json, 'r', encoding='utf-8') as f:
            for row in f:
                row = json.loads(row)

                text = row['text']
                label = row['label']
                vector = row['vector'][0]
                vector = vector / np.linalg.norm(vector, 2)

                vector_list.append(vector)

                text_info_list.append({
                    'text': text,
                    'label': label,
                })

                if count % 10000 == 0:
                    logger.info('[init index] count: {}'.format(count))

                count += 1

        logger.info('[init index] vector_list length: {}'.format(len(vector_list)))
        vector_list = np.array(vector_list, dtype=np.float32)
        self.index.add(vector_list)
        self.vector_list = vector_list
        self.text_info_list = text_info_list
        logger.info('[init index] finish')

    def sim_score(self, vector1, vector2, sim_mode='cosine'):
        if sim_mode == 'cosine':
            vector1 = vector1 / np.linalg.norm(vector1, 2)
            vector2 = vector2 / np.linalg.norm(vector2, 2)
            sim = np.sum(vector1 * vector2, axis=-1)
        elif sim_mode == 'probs':
            sim = np.sum(np.sqrt(vector1 + 1e-7) * np.sqrt(vector2 + 1e-7), axis=-1)
        else:
            vector1 = vector1 / np.linalg.norm(vector1, 2)
            vector2 = vector2 / np.linalg.norm(vector2, 2)
            sim = np.sum(vector1 * vector2, axis=-1)
        return sim

    def retrieval(self, vector: np.ndarray):
        vector = np.array([vector], dtype=np.float32)
        D, I = self.index.search(vector, self.top_k)

        result = list()
        for idx in I[0]:
            text_info = self.text_info_list[idx]
            idx_vector = self.vector_list[idx]

            sim = self.sim_score(
                vector1=vector,
                vector2=np.array([idx_vector], dtype=np.float32),
                sim_mode=self.sim_mode,
            )

            result.append({
                'text': text_info['text'],
                'label': text_info['label'],
                'score': round(float(sim), 4),
            })
        return result


def main():
    args = get_args()

    model_name = args.pretrained_model_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocabulary = Vocabulary.from_files(args.vocabulary)

    tokenizer = PretrainedBertTokenizer(model_name)

    # dataset
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

    # faiss
    faiss_index = FaissRetrieval(
        all_vector_json=args.all_vector,
        embedding_dim=args.n_clusters
    )

    while True:
        text = input('text input: ')
        text = str(text).strip()

        sample = {
            'tokens': tokenizer.tokenize(text),
            'label': '无关领域'
        }

        input_ids, targets = collate_fn([sample])
        input_ids = input_ids.to(device)
        with torch.no_grad():
            logits = model.forward(input_ids)

        logits = logits.detach().cpu().numpy()
        vector = logits[0]
        candidates: List[dict] = faiss_index.retrieval(vector)
        candidates = list(sorted(candidates, key=lambda x: x['score'], reverse=True))
        for candidate in candidates:
            text_ = candidate['text']
            label_ = candidate['label']
            score_ = candidate['score']

            logger.info('text: {}, label: {}, score: {}'.format(text_, label_, score_))


if __name__ == '__main__':
    main()
