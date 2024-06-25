#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from glob import glob
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.data.dataloader import DataLoader

from project_settings import project_path
from toolbox.torch.utils.data.dataset.wave_classifier_excel_dataset import WaveClassifierExcelDataset
from toolbox.torch.utils.data.vocabulary import Vocabulary
from toolbox.torchaudio.models.speaker_identification.cnn_text_dependent import CnnTextDependentWaveClassifier
from toolbox.torch.modules.loss import FocalLoss
from toolbox.torch.training.metrics.verbose_categorical_accuracy import CategoricalAccuracyVerbose


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./', type=str)
    parser.add_argument('--ckpt_path', required=True, type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--output_filename', default='evaluation.xlsx', type=str)
    parser.add_argument('--full_dataset', default='dataset.xlsx', type=str)
    parser.add_argument('--train_dataset', default='train.xlsx', type=str)
    parser.add_argument('--test_dataset', default='test.xlsx', type=str)

    args = parser.parse_args()
    return args


args = get_args()
file_dir = Path(args.file_dir)
file_dir.mkdir(parents=True, exist_ok=True)


vocabulary = Vocabulary.from_files(file_dir / 'vocabulary')


class CollateFunction(object):
    def __init__(self):
        pass

    def __call__(self, batch: List[dict]):
        array_list = list()
        label_list = list()
        for sample in batch:
            array = sample['waveform']
            label = sample['label']

            array_list.append(array)
            label_list.append(label)

        array_list = torch.stack(array_list)
        label_list = torch.stack(label_list)
        return array_list, label_list


collate_fn = CollateFunction()


train_dataset = WaveClassifierExcelDataset(
    vocab=vocabulary,
    excel_file=file_dir / args.train_dataset,
    expected_sample_rate=8000,
    max_wave_value=32768.0,
)
train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    # Linux 系统中可以使用多个子进程加载数据, 而在 Windows 系统中不能.
    num_workers=0 if platform.system() == 'Windows' else os.cpu_count(),
    collate_fn=collate_fn,
    pin_memory=False,
    # prefetch_factor=64,
)

test_dataset = WaveClassifierExcelDataset(
    vocab=vocabulary,
    excel_file=file_dir / args.test_dataset,
    expected_sample_rate=8000,
    max_wave_value=32768.0,
)
test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0 if platform.system() == 'Windows' else os.cpu_count(),
    collate_fn=collate_fn,
    pin_memory=False,
    # prefetch_factor=64,
)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        num_labels = vocabulary.get_vocab_size(namespace='labels')

        self.model = CnnTextDependentWaveClassifier(
            num_labels=num_labels,
            conv1d_block_param_list=[
                {
                    'batch_norm': True,
                    'in_channels': 80,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
                {
                    # 'batch_norm': True,
                    'in_channels': 16,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
                {
                    # 'batch_norm': True,
                    'in_channels': 16,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
            ],
            feedforward_param={
                'input_dim': 16,
                'num_layers': 2,
                'hidden_dims': 32,
                'activations': 'relu',
                'dropout': 0.1,
            },
            mel_spectrogram_param={
                'sample_rate': 8000,
                'n_fft': 512,
                'win_length': 200,
                'hop_length': 80,
                'f_min': 10,
                'f_max': 3800,
                'window_fn': 'hamming',
                'n_mels': 80,
            }
        )
        self._accuracy = CategoricalAccuracyVerbose(
            index_to_token=vocabulary.get_index_to_token_vocabulary('labels'),
        )
        self._loss = FocalLoss(
            num_classes=num_labels,
        )

    def forward(self, inputs: torch.Tensor, labels: torch.LongTensor = None):
        logits = self.model.forward(inputs)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if labels is not None:
            loss = self._loss(logits, labels.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, labels)

        return output_dict

    def train_dataloader(self):
        return train_data_loader

    def val_dataloader(self):
        return test_data_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)

        result = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler
            },
        }
        return result

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self.forward(data, labels)

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
        data, labels = batch
        outputs = self.forward(data, labels)
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
        data, labels = batch
        outputs = self.forward(data, labels)
        return outputs

    def test_step_end(self, outputs):
        accuracy = self._accuracy.get_metric()
        for k, v in accuracy.items():
            self.log(k, v)
        return outputs


model = Model()
if args.ckpt_path is not None:
    model = model.load_from_checkpoint(
        file_dir / args.ckpt_path,
        map_location=torch.device('cpu')
    )
model.eval()


def export_state_dict():
    torch.save(model.state_dict(), file_dir / 'pytorch_model.bin')
    return


def export_jit():
    filename = os.path.join(project_path, 'datasets/test/0a4a881a-d686-4aaa-ac65-074ffb77f08a_en-US_1661709365.751234.wav')
    inputs = train_dataset.filename_to_waveform(filename)
    inputs = torch.unsqueeze(inputs, dim=0)
    example_inputs = (inputs,)

    # outputs = model.forward(inputs)

    # 模型序列化
    trace_model = torch.jit.trace(func=model, example_inputs=example_inputs, strict=False)
    trace_model.save(file_dir / 'cnn_voicemail.pth')

    # trace 方式. 将模型运行一遍, 以记录对张量的操作并生成图模型.
    trace_model = torch.jit.trace(func=model.model, example_inputs=example_inputs, strict=False)
    trace_model.save(file_dir / 'trace_model.zip')

    # 量化
    quantized_model = torch.quantization.quantize_dynamic(
        model.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    trace_quant_model = torch.jit.trace(func=quantized_model, example_inputs=example_inputs, strict=False)
    trace_quant_model.save(file_dir / 'trace_quant_model.zip')

    # script 方式. 通过解析代码来生成图模型, 相较于 trace 方式, 它可以处理 if 条件判断的情况.
    script_model = torch.jit.script(obj=model.model)
    script_model.save(file_dir / 'script_model.zip')

    # 量化
    quantized_model = torch.quantization.quantize_dynamic(
        model.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    script_quant_model = torch.jit.script(quantized_model)
    script_quant_model.save(file_dir / 'script_quant_model.zip')
    return


def export_onnx():
    """torch.stft 不能转 onnx"""
    filename = os.path.join(project_path, 'datasets/test/0a4a881a-d686-4aaa-ac65-074ffb77f08a_en-US_1661709365.751234.wav')
    inputs = train_dataset.filename_to_waveform(filename)
    inputs = torch.unsqueeze(inputs, dim=0)
    example_inputs = (inputs,)

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
