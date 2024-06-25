#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from python_speech_features import sigproc


def calc_energy(signal, samplerate=16000, winlen=0.025, winstep=0.01):
    """
    任意信号都可以看作是在电阻R=1 的电路上的电流 I. 则能量为 I^2
    """
    signal = np.array(signal, dtype=np.float32)
    power = np.square(signal)

    # 分帧
    frames = sigproc.framesig(power, winlen*samplerate, winstep*samplerate)
    # 各帧能量总和.
    energy = np.mean(frames, axis=-1)
    return energy


def calc_zero_crossing_rate(signal, samplerate=16000, winlen=0.025, winstep=0.01):
    """过零率. """
    signal = np.where(signal >= 0, 1, -1)
    cross_zero = np.where(signal[1:] != signal[:-1], 1, 0)

    frames = sigproc.framesig(cross_zero, winlen*samplerate, winstep*samplerate)
    _, n = frames.shape
    cross_zero_rate = np.mean(frames, axis=-1)

    return cross_zero_rate


def detect_silence(signal, samplerate=16000, winlen=0.025, winstep=0.01, min_energy=0.01, min_cross_zero_rate=0.05):
    """静音段检测"""
    energy = calc_energy(
        signal=signal,
        samplerate=samplerate,
        winlen=winlen,
        winstep=winstep,
    )
    cross_zero_rate = calc_zero_crossing_rate(
        signal=signal,
        samplerate=samplerate,
        winlen=winlen,
        winstep=winstep,
    )
    energy = energy < min_energy
    cross_zero_rate = cross_zero_rate < min_cross_zero_rate
    silence_signal = np.array(energy + cross_zero_rate, dtype=np.bool)
    silence_signal = silence_signal.tolist()

    frame_len = int(sigproc.round_half_up(winlen*samplerate))
    frame_step = int(sigproc.round_half_up(winstep*samplerate))

    silence_list = list()
    last_s = False
    for idx, s in enumerate(silence_signal):
        if s is True:
            if last_s is True:
                silence = silence_list.pop(-1)
                begin = silence[0]
                count = silence[1]
                silence_list.append([begin, count + 1])
            else:
                begin = frame_step * idx
                silence_list.append([begin, 1])

        last_s = s

    result = list()
    for silence in silence_list:
        begin = silence[0]
        count = silence[1]
        end = begin + frame_step * (count - 1) + frame_len
        result.append([begin, end])

    return result


if __name__ == '__main__':
    pass
