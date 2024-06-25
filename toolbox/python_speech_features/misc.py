#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import cv2 as cv
import numpy as np
from python_speech_features import sigproc
from python_speech_features import mfcc
from sklearn import preprocessing


def wave2spectrum(sample_rate, wave, winlen=0.025, winstep=0.01, nfft=512):
    """计算功率谱图像"""
    frames = sigproc.framesig(
        sig=wave,
        frame_len=winlen * sample_rate,
        frame_step=winstep * sample_rate,
        winfunc=np.hamming
    )
    spectrum = sigproc.powspec(
        frames=frames,
        NFFT=nfft
    )
    spectrum = spectrum.T
    return spectrum


def wave2spectrum_image(
    wave, sample_rate,
    xmax=10, xmin=-50,
    winlen=0.025, winstep=0.01, nfft=512,
    n_low_freq=None
):
    """
    :return: numpy.ndarray, shape=(time_step, n_dim)
    """
    spectrum = wave2spectrum(
        sample_rate, wave,
        winlen=winlen,
        winstep=winstep,
        nfft=nfft,
    )
    spectrum = np.log(spectrum, out=np.zeros_like(spectrum), where=(spectrum != 0))
    spectrum = spectrum.T
    gray = 255 * (spectrum - xmin) / (xmax - xmin)
    gray = np.array(gray, dtype=np.uint8)
    if n_low_freq is not None:
        gray = gray[:, :n_low_freq]

    return gray


def compute_delta(specgram: np.ndarray, win_length: int = 5):
    """
    :param specgram: shape=[time_steps, n_mels]
    :param win_length:
    :return:
    """
    n = (win_length - 1) // 2

    specgram = np.array(specgram, dtype=np.float32)

    kernel = np.arange(-n, n + 1, 1)
    kernel = np.reshape(kernel, newshape=(2 * n + 1, 1))
    kernel = np.array(kernel, dtype=np.float32) / 10

    delta = cv.filter2D(
        src=specgram,
        ddepth=cv.CV_32F,
        kernel=kernel,
    )
    return delta


def delta_mfcc_feature(signal, sample_rate):
    """
    为 GMM UBM 声纹识别模型, 编写此代码.

    https://github.com/pventrella20/Speaker_identification_-GMM-UBM-
    https://github.com/MChamith/Speaker_verification_gmm_ubm

    :param signal: np.ndarray
    :param sample_rate: frequenza del file audio
    :return:
    """

    # shape=[time_steps, n_mels]
    mfcc_feat = mfcc(
        signal=signal,
        samplerate=sample_rate,
        winlen=0.025,
        winstep=0.01,
        numcep=20,
        appendEnergy=True
    )

    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = compute_delta(mfcc_feat)
    combined = np.hstack(tup=(mfcc_feat, delta))
    return combined


if __name__ == '__main__':
    pass
