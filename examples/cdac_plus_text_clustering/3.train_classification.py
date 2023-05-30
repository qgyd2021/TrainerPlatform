#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import pickle
import platform
import queue
import random
import sys
from typing import List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import numpy as np
from pytorch_pretrained_bert.optimization import BertAdam
import torch
from torch.nn import functional
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange

from project_settings import project_path
from toolbox.torch.modules.loss import FocalLoss, HingeLoss, HingeLinear
from toolbox.torch.utils.data.dataset.text_classifier_json_dataset import TextClassifierJsonDataset
from toolbox.torch.utils.data.vocabulary import Vocabulary
from toolbox.torch.utils.data.tokenizers.pretrained_bert_tokenizer import PretrainedBertTokenizer
from toolbox.torchtext.models.text_clustering.cdac_plus import BertForConstrainClustering, CDACPlus
from toolbox.torchtext.models.text_clustering.utils import clustering_score


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_dir', default='chinese-bert-wwm-ext', type=str)

    parser.add_argument('--train_labeled', default='train_labeled.json', type=str)
    parser.add_argument('--valid_labeled', default='valid_labeled.json', type=str)
    parser.add_argument('--train_all', default='train_all.json', type=str)

    parser.add_argument('--vocabulary', default='vocabulary', type=str)

    parser.add_argument('--n_clusters', default=200, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--num_serialized_models_to_keep', default=10, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--serialization_dir', default='classification', type=str)
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()
    return args


def logging_config(file_dir: str):
    format = '[%(asctime)s] %(levelname)s \t [%(filename)s %(lineno)d] %(message)s'
    logging.basicConfig(format=format,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG)
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(file_dir, 'log.log'),
        encoding='utf-8',
        when='D',
        interval=1,
        backupCount=7
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(format))
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)

    return logger


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
    os.makedirs(args.serialization_dir, exist_ok=False)

    logger = logging_config(args.serialization_dir)

    model_name = args.pretrained_model_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vocabulary = Vocabulary.from_files(args.vocabulary)
    num_labels = vocabulary.get_vocab_size(namespace='labels')

    collate_fn = CollateFunction(
        vocab=vocabulary,
        token_min_padding_length=5,
    )

    train_labeled_dataset = TextClassifierJsonDataset(
        json_file=args.train_labeled,
        tokenizer=PretrainedBertTokenizer(model_name),
    )
    train_labeled_data_loader = DataLoader(
        dataset=train_labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if platform.system() == 'Windows' else os.cpu_count(),
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=2,
    )

    valid_labeled_dataset = TextClassifierJsonDataset(
        json_file=args.valid_labeled,
        tokenizer=PretrainedBertTokenizer(model_name),
    )
    valid_labeled_data_loader = DataLoader(
        dataset=valid_labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if platform.system() == 'Windows' else os.cpu_count(),
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=2,
    )
    # for batch in train_data_loader:
    #     print(batch)

    # Freezing all transformer (except the last layer)
    bert_for_constrain_clustering = BertForConstrainClustering.from_pretrained(model_name)
    for name, param in bert_for_constrain_clustering.bert.named_parameters():
        param.requires_grad = False
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True

    model = CDACPlus(
        backbone=bert_for_constrain_clustering,
        hidden_size=768,
        dropout=0.1,
        n_clusters=args.n_clusters,
        positive_weight=10,
    )
    model.to(device)

    optimizer = BertAdam(
        model.parameters(),
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        t_total=50000
    )

    best_model = None
    best_accuracy = None
    patience_count = 0
    global_step = 0

    model_filename_list = list()
    for idx_epoch in range(args.num_epochs):
        model.train()

        # training
        total_loss = 0
        total_examples, total_steps = 0, 0
        loss = 0
        for step, batch in enumerate(tqdm(train_labeled_data_loader, desc='Epoch={} (training)'.format(idx_epoch))):
            input_ids, label_ids = batch
            input_ids = input_ids.to(device)
            label_ids: torch.LongTensor = label_ids.to(device).long()

            logits = model.forward(input_ids)
            loss = model.focal_loss.forward(logits, label_ids.view(-1))
            model.accuracy(logits, label_ids)

            loss.backward()

            total_loss += loss.item()
            total_examples += input_ids.size(0)
            total_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        training_loss = total_loss / total_steps
        training_loss = round(training_loss, 4)
        training_accuracy = model.accuracy.get_metric(reset=True)['accuracy']
        training_accuracy = round(training_accuracy, 4)
        logger.info('Epoch: {}; training_loss: {}; training_accuracy: {}'.format(
            idx_epoch, training_loss, training_accuracy
        ))

        # evaluation
        model.eval()
        total_loss = 0
        total_examples, total_steps = 0, 0
        for step, batch in enumerate(tqdm(valid_labeled_data_loader, desc='Epoch={} (evaluation)'.format(idx_epoch))):
            input_ids, label_ids = batch
            input_ids = input_ids.to(device)
            label_ids: torch.LongTensor = label_ids.to(device).long()

            with torch.no_grad():
                logits = model.forward(input_ids)
                loss = model.focal_loss.forward(logits, label_ids.view(-1))
                model.accuracy(logits, label_ids)

            total_loss += loss.item()
            total_examples += input_ids.size(0)
            total_steps += 1

            global_step += 1

        evaluation_loss = total_loss / total_steps
        evaluation_loss = round(evaluation_loss, 4)
        evaluation_accuracy = model.accuracy.get_metric(reset=True)['accuracy']
        evaluation_accuracy = round(evaluation_accuracy, 4)
        logger.info('Epoch: {}; evaluation_loss: {}; evaluation_accuracy: {}'.format(
            idx_epoch, evaluation_loss, evaluation_accuracy
        ))

        metrics = {
            'training_loss': training_loss,
            'training_accuracy': training_accuracy,
            'evaluation_loss': evaluation_loss,
            'evaluation_accuracy': evaluation_accuracy,
        }
        metrics_filename = os.path.join(args.serialization_dir, 'metrics_epoch_{}.json'.format(idx_epoch))
        with open(metrics_filename, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        model_filename = os.path.join(args.serialization_dir, 'pretrain_epoch_{}.bin'.format(idx_epoch))
        model_filename_list.append(model_filename)
        if len(model_filename_list) >= args.num_serialized_models_to_keep:
            model_filename_to_delete = model_filename_list.pop(0)
            os.remove(model_filename_to_delete)
        torch.save(model.state_dict(), model_filename)

        # early stop
        if best_model is None or best_accuracy is None:
            best_model = copy.deepcopy(model)
            best_accuracy = evaluation_accuracy
            model_filename = os.path.join(args.serialization_dir, 'best.bin')
            torch.save(model.state_dict(), model_filename)
        elif evaluation_accuracy > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = evaluation_accuracy
            model_filename = os.path.join(args.serialization_dir, 'best.bin')
            torch.save(model.state_dict(), model_filename)
            patience_count = 0
        elif patience_count >= args.patience:
            break
        else:
            patience_count += 1

    return


if __name__ == '__main__':
    main()
