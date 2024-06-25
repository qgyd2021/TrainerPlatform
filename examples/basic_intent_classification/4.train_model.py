#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import pickle
import platform
import sys
from typing import List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch.utils.data.dataloader import DataLoader

from toolbox.torch.utils.data.dataset.text_classifier_json_dataset import HierarchicalClassificationJsonDataset
from toolbox.torch.utils.data.vocabulary import Vocabulary
from toolbox.torch.modules.loss import FocalLoss, NegativeEntropy
from toolbox.torch.training.metrics.categorical_accuracy import CategoricalAccuracy
from toolbox.torch.utils.data.tokenizers.pretrained_bert_tokenizer import PretrainedBertTokenizer
from toolbox.torch.nn.functional.hierarchical_softmax import HierarchicalSoftMax
from toolbox.torchtext.models.text_classification.text_cnn import TextCNN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./', type=str)

    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--pretrained_model_dir', required=True, type=str)

    parser.add_argument('--hierarchical_labels_pkl', default='hierarchical_labels.pkl', type=str)
    parser.add_argument('--vocabulary', default='vocabulary', type=str)

    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='valid.json', type=str)

    args = parser.parse_args()
    return args


args = get_args()
ckpt_path = args.ckpt_path
pretrained_model_dir = args.pretrained_model_dir

file_dir = Path(args.file_dir)
file_dir.mkdir(exist_ok=True)


vocabulary = Vocabulary.from_files(args.vocabulary)


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


collate_fn = CollateFunction(vocab=vocabulary, token_min_padding_length=5)


tokenizer = PretrainedBertTokenizer(pretrained_model_dir)

train_dataset = HierarchicalClassificationJsonDataset(
    json_file=args.train_subset,
    tokenizer=tokenizer,
    n_hierarchical=2,
)
train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    # Linux 系统中可以使用多个子进程加载数据, 而在 Windows 系统中不能.
    num_workers=0 if platform.system() == 'Windows' else os.cpu_count(),
    collate_fn=collate_fn,
    pin_memory=False,
    prefetch_factor=2,
)

test_dataset = HierarchicalClassificationJsonDataset(
    json_file=args.valid_subset,
    tokenizer=tokenizer,
    n_hierarchical=2,
)
test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0 if platform.system() == 'Windows' else os.cpu_count(),
    collate_fn=collate_fn,
    pin_memory=False,
    prefetch_factor=2,
)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        num_labels = vocabulary.get_vocab_size(namespace='labels')
        self.model = TextCNN(
            num_embeddings=vocabulary.get_vocab_size(namespace='tokens'),
            embedding_dim=128,
            stacked_self_attention_encoder_param={
                'input_dim': 128,
                'hidden_dim': 128,
                'projection_dim': 128,
                'feedforward_hidden_dim': 128,
                'num_layers': 2,
                'num_attention_heads': 4,
                'use_positional_encoding': False,
            },
            cnn_encoder_param={
                'input_dim': 128,
                'num_filters': 32,
                'ngram_filter_sizes': [2, 3, 4, 5],
                'output_dim': 128,
            },
            output_dim=128,
        )

        with open(args.hierarchical_labels_pkl, 'rb') as f:
            hierarchical_labels = pickle.load(f)
        self.hierarchical_softmax = HierarchicalSoftMax(
            classifier_input_dim=128,
            hierarchical_labels=hierarchical_labels,
            activation='softmax',
        )

        self._accuracy = CategoricalAccuracy()
        self._loss = FocalLoss(
            num_classes=num_labels,
            inputs_logits=False,
        )

    def forward(self, inputs: torch.LongTensor, label: torch.LongTensor = None):
        logits = self.model.forward(inputs)

        probs = self.hierarchical_softmax(logits)

        output_dict = {'probs': probs}

        if label is not None:
            loss = self._loss(probs, label.long().view(-1))
            output_dict['loss'] = loss
            self._accuracy(probs, label)

        return output_dict

    def train_dataloader(self):
        return train_data_loader

    def val_dataloader(self):
        return test_data_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)

        result = {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': lr_scheduler
            # },
        }
        return result

    def training_step(self, batch, batch_idx):
        data, label = batch
        outputs = self.forward(data, label)

        accuracy = self._accuracy.get_metric()
        for k, v in accuracy.items():
            self.log(k, v, prog_bar=True)
        return outputs

    def training_epoch_end(self, outputs):
        accuracy = self._accuracy.get_metric(reset=True)
        for k, v in accuracy.items():
            self.log(k, v, prog_bar=True)
        return None

    def validation_step(self, batch, batch_idx):
        data, label = batch
        outputs = self.forward(data, label)
        return outputs

    def validation_step_end(self, outputs):
        accuracy = self._accuracy.get_metric()
        for k, v in accuracy.items():
            self.log('val_{}'.format(k), v, prog_bar=True)
        return outputs

    def validation_epoch_end(self, outputs):
        accuracy = self._accuracy.get_metric(reset=True)
        for k, v in accuracy.items():
            self.log('val_{}'.format(k), v, prog_bar=True)
        return outputs

    def test_step(self, batch, batch_idx):
        data, label = batch
        outputs = self.forward(data, label)
        return outputs

    def test_step_end(self, outputs):
        accuracy = self._accuracy.get_metric()
        for k, v in accuracy.items():
            self.log(k, v)
        return outputs


model = Model()
if args.ckpt_path is not None:
    model = model.load_from_checkpoint(
        args.ckpt_path,
        map_location=torch.device('cpu')
    )
model.train()

print(model)


ckpt_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_accuracy',
    save_top_k=10,
    mode='max',
)

early_stopping = pl.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    mode='max',
)

trainer = Trainer(
    max_epochs=200,
    callbacks=[ckpt_callback, early_stopping],
    weights_summary='full',
    progress_bar_refresh_rate=10,
    profiler='simple',
    accumulate_grad_batches=1,
    default_root_dir=file_dir,

    # https://mp.weixin.qq.com/s?__biz=MzI1MjQ2OTQ3Ng==&mid=2247561650&idx=1&sn=ea6de6d2a6e4831c735d98d37cbfd026&chksm
    gpus=[0] if torch.cuda.is_available() else None,
)

trainer.fit(
    model=model,
)


if __name__ == '__main__':
    pass
