#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

import matplotlib.pyplot as plt
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


def main():
    args = get_args()

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=8000,
        n_fft=512,
        win_length=200,
        hop_length=80,
        f_min=10,
        f_max=3800,
        window_fn=torch.hamming_window,
        n_mels=80,
    )

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=512,
        win_length=200,
        hop_length=80,
        window_fn=torch.hamming_window,
    )

    sample_rate, signal = wavfile.read(args.wav_file)

    max_wave_value = 32768.0
    signal = signal / max_wave_value

    x = torch.tensor(signal, dtype=torch.float32)
    x = spectrogram(x) + 1e-6

    x = x.numpy()
    x = np.log(x, out=np.zeros_like(x), where=(x != 0))

    xmin = np.min(x)
    xmax = np.max(x)
    xmin = -50
    xmax = 10
    print("xmax: {}; xmin: {}".format(xmax, xmin))

    gray = 255 * (x - xmin) / (xmax - xmin)
    gray = np.array(gray, dtype=np.uint8)
    show_image(gray)

    # plt.imshow(gray)
    # plt.show()
    return


if __name__ == "__main__":
    main()
