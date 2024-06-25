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
import torch
from torch.utils.data.dataloader import DataLoader

from project_settings import project_path
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
    shuffle=True,
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


model = Model()
if args.ckpt_path is not None:
    model = model.load_from_checkpoint(
        args.ckpt_path,
        map_location=torch.device('cpu')
    )
model.eval()

print(model)


def export_state_dict():
    torch.save(model.state_dict(), file_dir / 'pytorch_model.bin')
    return


def export_jit():
    text = '导出序列化模型.'
    instance = train_dataset.text_to_instance(text)
    instance['labels'] = '无关领域_无关领域'

    batch_tokens, _ = collate_fn([instance])
    example_inputs = (batch_tokens,)

    # 模型序列化
    # trace 方式. 将模型运行一遍, 以记录对张量的操作并生成图模型.
    trace_model = torch.jit.trace(func=model, example_inputs=example_inputs, strict=False)
    trace_model.save(file_dir / 'trace_model.zip')

    # 量化
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    trace_quant_model = torch.jit.trace(func=quantized_model, example_inputs=example_inputs, strict=False)
    trace_quant_model.save(file_dir / 'trace_quant_model.zip')

    # script 方式. 通过解析代码来生成图模型, 相较于 trace 方式, 它可以处理 if 条件判断的情况.
    # script_model = torch.jit.script(obj=model)
    # script_model.save(file_dir / 'script_model.zip')

    # 量化
    # quantized_model = torch.quantization.quantize_dynamic(
    #     model, {torch.nn.Linear}, dtype=torch.qint8
    # )
    # script_quant_model = torch.jit.script(quantized_model)
    # script_quant_model.save(file_dir / 'script_quant_model.zip')
    return


def export_onnx():
    text = '导出序列化模型.'
    instance = train_dataset.text_to_instance(text)
    instance['labels'] = '无关领域_无关领域'

    batch_tokens, _ = collate_fn([instance])
    example_inputs = (batch_tokens,)

    trace_model = torch.jit.load(file_dir / 'trace_model.zip')

    # torch.onnx.export 默认使用 trace 模式
    # 转换为 onnx 模型
    torch.onnx.export(
        model=trace_model,
        args=example_inputs,
        f=file_dir / 'trace_model.onnx',
        input_names=["inputs"],
        output_names=["outputs"],
    )
    return


def main():
    export_state_dict()
    export_jit()
    # export_onnx()
    return


if __name__ == '__main__':
    main()
