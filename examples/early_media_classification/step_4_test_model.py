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
sys.path.append(os.path.join(pwd, "../../"))

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.data.dataloader import DataLoader
import torchaudio

from project_settings import project_path
from toolbox.torch.utils.data.dataset.wave_classifier_excel_dataset import WaveClassifierExcelDataset
from toolbox.torch.utils.data.vocabulary import Vocabulary
from toolbox.torchaudio.models.speaker_identification.cnn_text_dependent import CnnTextDependentWaveClassifier, CnnTextDependentSpectrumClassifier
from toolbox.torch.modules.loss import FocalLoss
from toolbox.torch.training.metrics.verbose_categorical_accuracy import CategoricalAccuracyVerbose


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", default="./", type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--output_filename", default="evaluation.xlsx", type=str)
    parser.add_argument("--full_dataset", default="dataset.xlsx", type=str)
    parser.add_argument("--train_dataset", default="train.xlsx", type=str)
    parser.add_argument("--test_dataset", default="test.xlsx", type=str)

    args = parser.parse_args()
    return args


args = get_args()
file_dir = Path(args.file_dir)
file_dir.mkdir(exist_ok=True)


vocabulary = Vocabulary.from_files(file_dir / "vocabulary")


class EarlyMediaTransform(object):
    def __init__(self,
                 n_fft: int = 512,
                 win_length: int = 200,
                 hop_length: int = 80,
                 xmin: int = -50,
                 xmax: int = 10,
                 n_low_freq: int = 100,
                 ):
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.hamming_window,
        )
        self.xmin = xmin
        self.xmax = xmax

        self.n_low_freq = n_low_freq

    def forward(self, inputs: torch.Tensor):
        x = self.spectrogram(inputs) + 1e-6

        if self.n_low_freq is not None:
            x = x[:self.n_low_freq]

        x = torch.log(x)

        x = (x - self.xmin) / (self.xmax - self.xmin)
        return x


class CollateFunction(object):
    def __init__(self):
        self.transform = EarlyMediaTransform()

    def __call__(self, batch: List[dict]):
        array_list = list()
        label_list = list()

        with torch.no_grad():
            for sample in batch:
                waveform = sample["waveform"]
                label = sample["label"]

                array = self.transform.forward(waveform)

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
    num_workers=0 if platform.system() == "Windows" else os.cpu_count(),
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
    num_workers=0 if platform.system() == "Windows" else os.cpu_count(),
    collate_fn=collate_fn,
    pin_memory=False,
    # prefetch_factor=64,
)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        num_labels = vocabulary.get_vocab_size(namespace="labels")

        self.model = CnnTextDependentSpectrumClassifier(
            num_labels=num_labels,
            conv2d_block_param_list=[
                {
                    # "batch_norm": True,
                    "in_channels": 1,
                    "out_channels": 4,
                    "kernel_size": 3,
                    "stride": 2,
                    # "padding": "same",
                    "dilation": 3,
                    "activation": "relu",
                    "dropout": 0.1,
                },
                {
                    # "batch_norm": True,
                    "in_channels": 4,
                    "out_channels": 4,
                    "kernel_size": 5,
                    "stride": 3,
                    # "padding": "same",
                    "dilation": 3,
                    "activation": "relu",
                    "dropout": 0.1,
                },
            ],
            feedforward_param={
                "input_dim": 48,
                "num_layers": 2,
                "hidden_dims": [32, 16],
                "activations": "relu",
                "dropout": 0.1,
            },
        )
        self._accuracy = CategoricalAccuracyVerbose(
            index_to_token=vocabulary.get_index_to_token_vocabulary("labels"),
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
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler
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
            self.log("val_{}".format(k), v, prog_bar=True)
        return outputs

    def validation_epoch_end(self, outputs):
        accuracy = self._accuracy.get_metric(reset=True)
        for k, v in accuracy.items():
            self.log("val_{}".format(k), v, prog_bar=True)
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
        map_location=torch.device("cpu")
    )
model.eval()
print(model)


def main():
    import pandas as pd
    from tqdm import tqdm

    index_to_token = vocabulary.get_index_to_token_vocabulary(namespace="labels")

    df = pd.read_excel(file_dir / args.full_dataset)

    result = list()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        filename = row["filename"]
        label = row["label"]
        label_int = vocabulary.get_token_index(label, namespace="label")

        waveform = train_dataset.filename_to_waveform(filename)

        inputs = [
            {
                "waveform": waveform,
                "label": torch.tensor(label_int, dtype=torch.int64),
            }
        ]
        inputs, _ = collate_fn(inputs)

        outputs = model.forward(inputs)
        probs = outputs["probs"]
        label_idx = torch.argmax(probs, dim=-1)
        label_idx = label_idx.numpy()
        label_str = index_to_token[label_idx[0]]

        prob = probs[0][label_idx].detach().numpy()

        row = dict(row)
        row["predict"] = label_str
        row["correct"] = 1 if label_str == label else 0
        row["prob"] = round(float(prob), 4)

        result.append(row)

    result = pd.DataFrame(result)
    result.to_excel(file_dir / args.output_filename, index=False, encoding="utf_8_sig")

    return


if __name__ == "__main__":
    main()
