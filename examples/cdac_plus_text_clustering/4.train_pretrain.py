#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
import itertools
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import pickle
import platform
import queue
import random
import sys
from typing import Any, Callable, Dict, TypeVar, Generic, Sequence, List, Optional, Sized

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import numpy as np
from pytorch_pretrained_bert.optimization import BertAdam
import torch
from torch.nn import functional
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import IterableDataset, Sampler, SequentialSampler, RandomSampler, BatchSampler, Dataset
from torch.utils.data.dataloader import DataLoader, T_co, T, _collate_fn_t, _worker_init_fn_t, _BaseDataLoaderIter
from torch.utils.data.dataset import Dataset

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
    parser.add_argument('--k_classes', default=14, type=int)
    parser.add_argument('--upper_threshold', default=0.95, type=float)
    parser.add_argument('--lower_threshold', default=0.455, type=float)
    parser.add_argument('--num_epochs', default=46, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--num_serialized_models_to_keep', default=10, type=int)
    parser.add_argument('--serialization_dir', default='pretrain', type=str)
    # ./classification/best.bin
    parser.add_argument('--pretrain_model_filename', default=None, type=str)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--with_classification_cross_training', default=False, type=bool)

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


class ClassDependentDataLoader(Generic[T_co]):
    """
    (1)从 dataset 中将样本按标签分开.
    (2)按各标签剩余样本数量为权重, 从类别中抽取 n 个类别.
    (3)从抽中的 n 个类别中, 按各类别中样本数量抽取样本.
    (4)组成 batch.
    (5)sampler
    """

    def __init__(self,
                 dataset: Dataset[T_co],
                 batch_size: Optional[int] = 1,
                 k_classes: int = 14,
                 shuffle: bool = False,
                 collate_fn: Optional[_collate_fn_t] = None,
                 ):
        super(ClassDependentDataLoader, self).__init__()
        self.dataset: Dataset[T_co] = dataset
        self.batch_size = batch_size
        self.k_classes = k_classes
        self.shuffle = shuffle
        self.collate_fn = collate_fn

        dataset: Sized
        self.total_batches = len(dataset) // batch_size

        self.label_to_index_list = None

    def split_dataset_by_label(self, dataset: Dataset[T_co]) -> Dict[Any, List[int]]:
        label_to_index_list: Dict[Any, List[int]] = dict()
        for idx, sample in enumerate(dataset.samples):
            label = sample['label']

            if label not in label_to_index_list.keys():
                label_to_index_list[label] = list()

            label_to_index_list[label].append(idx)

        if self.shuffle:
            for k, v in label_to_index_list.items():
                random.shuffle(v)

        return label_to_index_list

    @staticmethod
    def get_labels_and_weights(label_to_index_list: Dict[Any, List[int]]):
        label_to_count = dict()
        for k, v in label_to_index_list.items():
            label_to_count[k] = len(v)

        labels = list()
        weights = list()
        for k, v in label_to_count.items():
            labels.append(k)
            weights.append(v)
        return labels, weights

    @staticmethod
    def weighted_samples(population, k: int, weights: List[float] = None):
        """带权重无放回抽样"""
        result = list()

        population_ = copy.deepcopy(population)
        weights_ = copy.deepcopy(weights)

        for i in range(k):
            cum_weights = list(itertools.accumulate(weights_))
            pointer = random.random()
            pointer *= cum_weights[-1]
            for idx, cum_weight in enumerate(cum_weights):
                if pointer <= cum_weight:
                    weights_.pop(idx)
                    p = population_.pop(idx)
                    result.append(p)
                    break

        return result

    def __len__(self):
        return self.total_batches

    def __iter__(self) -> Any:
        self.label_to_index_list = self.split_dataset_by_label(self.dataset)

        for _ in range(self.total_batches):
            batch_samples = list()

            labels, weights = self.get_labels_and_weights(self.label_to_index_list)

            selected_labels = self.weighted_samples(labels, k=self.k_classes, weights=weights)
            selected_labels_samples_count = [len(self.label_to_index_list[selected_label]) for selected_label in
                                             selected_labels]

            total = sum(selected_labels_samples_count)
            if total < self.batch_size:
                break

            mini_total = 0
            for selected_label, sample_count in zip(selected_labels, selected_labels_samples_count):
                count = sample_count / total * self.batch_size
                count = int(round(count))
                # print(selected_label, count)

                for _ in range(count):
                    try:
                        index = self.label_to_index_list[selected_label].pop()
                    except IndexError:
                        break
                    batch_samples.append(index)
                    mini_total += 1
                    if mini_total >= self.batch_size:
                        break

            if mini_total < self.batch_size:
                label_count_list = list(zip(selected_labels, selected_labels_samples_count))
                label_count_list = sorted(label_count_list, key=lambda x: x[1])

                for label, _ in label_count_list:
                    while True:
                        try:
                            index = self.label_to_index_list[label].pop()
                        except IndexError:
                            break
                        batch_samples.append(index)
                        mini_total += 1
                        if mini_total >= self.batch_size:
                            break
                    if mini_total >= self.batch_size:
                        break

            batch = list()
            for index in batch_samples:
                instance = self.dataset[index]
                batch.append(instance)

            random.shuffle(batch)

            batch = self.collate_fn(batch)
            yield batch


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

    upper_threshold = args.upper_threshold
    lower_threshold = args.lower_threshold
    eta = 1.1 * 0.009

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vocabulary = Vocabulary.from_files(args.vocabulary)
    num_labels = vocabulary.get_vocab_size(namespace='labels')

    collate_fn = CollateFunction(
        vocab=vocabulary,
        token_min_padding_length=5,
    )

    train_all_dataset = TextClassifierJsonDataset(
        json_file=args.train_all,
        tokenizer=PretrainedBertTokenizer(model_name),
    )
    train_all_data_loader = ClassDependentDataLoader(
        dataset=train_all_dataset,
        batch_size=args.batch_size,
        k_classes=args.k_classes,
        shuffle=True,
        collate_fn=collate_fn,
    )

    train_labeled_dataset = TextClassifierJsonDataset(
        json_file=args.train_labeled,
        tokenizer=PretrainedBertTokenizer(model_name),
    )
    train_labeled_data_loader = ClassDependentDataLoader(
        dataset=train_labeled_dataset,
        batch_size=args.batch_size,
        k_classes=args.k_classes,
        shuffle=True,
        collate_fn=collate_fn,
    )
    train_labeled_classification_data_loader = DataLoader(
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
    valid_labeled_data_loader = ClassDependentDataLoader(
        dataset=valid_labeled_dataset,
        batch_size=args.batch_size,
        k_classes=args.k_classes,
        shuffle=True,
        collate_fn=collate_fn,
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
        positive_weight=5,
    )
    if args.pretrain_model_filename is not None:
        with open(args.pretrain_model_filename, 'rb') as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = int(len(train_all_dataset) / args.batch_size) * args.num_epochs * 3
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        t_total=t_total
    )

    global_step = 0
    model_filename_list = list()
    for idx_epoch in range(args.num_epochs):
        model.train()

        # classification training
        classification_loss = 0.0
        classification_accuracy = 0.0
        if args.with_classification_cross_training:
            total_loss = 0
            total_examples, total_steps = 0, 0
            for step, batch in enumerate(tqdm(train_labeled_classification_data_loader, desc='Epoch={} (classification)'.format(idx_epoch))):
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
            classification_loss = total_loss / total_steps
            classification_loss = round(classification_loss, 4)
            classification_accuracy = model.accuracy.get_metric(reset=True)['accuracy']
            classification_accuracy = round(classification_accuracy, 4)
            logger.info('Epoch: {}; classification_loss: {}; classification_accuracy: {}'.format(
                idx_epoch, classification_loss, classification_accuracy
            ))

        # supervised training
        total_loss = 0
        total_examples, total_steps = 0, 0
        y_pred = list()
        y_true = list()
        for step, batch in enumerate(tqdm(train_labeled_data_loader, desc='Epoch={} (supervised)'.format(idx_epoch))):
            input_ids, label_ids = batch
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            outputs = model.supervised_training(
                input_ids,
                label_ids,
                upper_threshold,
                lower_threshold,
            )
            loss = outputs['loss']

            # shape=[batch_size, num_labels]
            logits = model.forward(input_ids)

            # shape=[batch_size]
            pred = torch.argmax(logits, dim=1)

            y_pred.append(pred)
            y_true.append(label_ids)

            total_loss += loss.item()
            total_examples += input_ids.size(0)
            total_steps += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        y_pred = torch.hstack(y_pred)
        y_true = torch.hstack(y_true)

        scores = clustering_score(
            y_pred.detach().cpu().numpy(),
            y_true.detach().cpu().numpy(),
        )

        supervised_training_loss = total_loss / total_steps
        supervised_training_loss = round(supervised_training_loss, 4)

        log_str = 'Epoch: {}; supervised_training_loss: {}'.format(idx_epoch, supervised_training_loss)
        for k, v in scores.items():
            log_str += ', {}: {}'.format(k, v)
        logger.info(log_str)

        # semi-supervised training
        total_loss = 0
        total_examples, total_steps = 0, 0
        for step, batch in enumerate(tqdm(train_all_data_loader, desc='Epoch={} (semi-supervised)'.format(idx_epoch))):
            input_ids, label_ids = batch
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            outputs = model.semi_supervised_training(
                input_ids,
                label_ids,
                upper_threshold,
                lower_threshold,
                unknown_label=vocabulary.get_token_index(
                    '无关领域',
                    namespace='labels',
                )
            )
            loss = outputs['loss']

            total_loss += loss.item()
            total_examples += input_ids.size(0)
            total_steps += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        semi_supervised_training_loss = total_loss / total_steps
        semi_supervised_training_loss = round(semi_supervised_training_loss, 4)
        logger.info('Epoch: {}; semi_supervised_training_loss: {}'.format(idx_epoch, semi_supervised_training_loss))

        # evaluation
        model.eval()
        total_loss = 0
        total_examples, total_steps = 0, 0
        y_pred = list()
        y_true = list()
        for step, batch in enumerate(tqdm(valid_labeled_data_loader, desc='Epoch={} (evaluation)'.format(idx_epoch))):
            input_ids, label_ids = batch
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            with torch.no_grad():
                outputs = model.supervised_training(
                    input_ids,
                    label_ids,
                    upper_threshold,
                    lower_threshold,
                )
                loss = outputs['loss']

                # shape=[batch_size, num_labels]
                logits = model.forward(input_ids)

                # shape=[batch_size]
                pred = torch.argmax(logits, dim=1)

                y_pred.append(pred)
                y_true.append(label_ids)

            total_loss += loss.item()
            total_examples += input_ids.size(0)
            total_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        y_pred = torch.hstack(y_pred)
        y_true = torch.hstack(y_true)

        scores = clustering_score(
            y_pred.detach().cpu().numpy(),
            y_true.detach().cpu().numpy(),
        )

        evaluation_loss = total_loss / total_steps
        evaluation_loss = round(evaluation_loss, 4)

        log_str = 'Epoch: {}; evaluation_loss: {}'.format(idx_epoch, evaluation_loss)
        for k, v in scores.items():
            log_str += ', {}: {}'.format(k, v)
        logger.info(log_str)

        metrics = {
            'upper_threshold': round(upper_threshold, 4),
            'lower_threshold': round(lower_threshold, 4),
            'eta': round(eta, 4),
            'classification_loss': classification_loss,
            'classification_accuracy': classification_accuracy,
            'supervised_training_loss': supervised_training_loss,
            'semi_supervised_training_loss': semi_supervised_training_loss,
            'evaluation_loss': evaluation_loss,
            **scores,
        }
        metrics_filename = os.path.join(args.serialization_dir, 'metrics_epoch_{}.json'.format(idx_epoch))
        with open(metrics_filename, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        model_filename = os.path.join(args.serialization_dir, 'pretrain_epoch_{}.bin'.format(idx_epoch))
        model_filename_list.append(model_filename)
        if len(model_filename_list) > args.num_serialized_models_to_keep:
            model_filename_to_delete = model_filename_list.pop(0)
            os.remove(model_filename_to_delete)
        torch.save(model.state_dict(), model_filename)

        model_filename = os.path.join(args.serialization_dir, 'best.bin')
        torch.save(model.state_dict(), model_filename)

        # update threshold
        upper_threshold -= eta
        lower_threshold += eta * 0.1
        if upper_threshold < lower_threshold:
            logger.info('upper threshold less than lower threshold; break.')
            break

    return


if __name__ == '__main__':
    main()
