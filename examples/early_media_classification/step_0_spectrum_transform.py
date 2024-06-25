#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

import numpy as np
from scipy.io import wavfile
import torch
import torchaudio

from toolbox.cv2.misc import show_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav_file",
        default=r"E:\programmer\asr_datasets\voicemail\wav_finished\EarlyMedia-62\wav_finished\voicemail\3300999628152207606_48000.wav",
        type=str
    )
    parser.add_argument("--task", default="default", type=str)
    parser.add_argument("--filename_patterns", nargs="+")

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


def main():
    args = get_args()

    transform = EarlyMediaTransform()

    sample_rate, signal = wavfile.read(args.wav_file)

    max_wave_value = 32768.0
    signal = signal / max_wave_value

    x = torch.tensor(signal, dtype=torch.float32)
    x = transform.forward(x)

    x = x.numpy()
    x = x * 255

    gray = np.array(x, dtype=np.uint8)
    show_image(gray)
    return


if __name__ == "__main__":
    main()
