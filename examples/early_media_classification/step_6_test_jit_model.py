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
from scipy.io import wavfile
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
    parser.add_argument(
        "--model_dir", 
        default=(project_path / "trained_models/early_media_20240417_2353").as_posix(), 
        type=str
    )
    args = parser.parse_args()
    return args


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


def main():
    args = get_args()
    model_dir = Path(args.model_dir)
    model_path = model_dir / "trace_model.zip"
    vocabulary_path = model_dir / "vocabulary"

    model = torch.jit.load(model_path)
    vocabulary = Vocabulary.from_files(vocabulary_path)

    collate_fn = CollateFunction()

    max_wave_value = 32768.0

    filename = os.path.join(project_path, "datasets/test/0a4a881a-d686-4aaa-ac65-074ffb77f08a_en-US_1661709365.751234.wav")
    sample_rate, waveform = wavfile.read(filename)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    waveform = waveform / max_wave_value

    inputs = [
        {
            "waveform": waveform,
            "label": torch.tensor(1, dtype=torch.int64),
        }
    ]
    inputs, _ = collate_fn(inputs)

    logits = model.forward(inputs)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    argmax = torch.argmax(probs, dim=-1)
    argmax = argmax.tolist()
    index = argmax[0]
    label = vocabulary.get_token_from_index(index, namespace="labels")
    print(label)

    return


if __name__ == "__main__":
    main()
