#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

from toolbox.python_speech_features.silence_detect import detect_silence


def calc_wave_features(signal, sample_rate):
    assert signal.dtype == np.int16
    assert sample_rate == 8000

    signal = np.array(signal, dtype=np.float32)
    # plt.plot(signal)
    # plt.show()

    l = len(signal)

    # 均值
    mean = np.mean(signal)

    # 方差
    var = np.var(signal)

    # 百分位数
    per = np.percentile(signal, q=[1, 25, 50, 75, 99])
    per1, per25, per50, per75, per99 = per

    # 静音段占比
    silences = detect_silence(
        signal=signal,
        samplerate=sample_rate,
        min_energy=120,
        min_cross_zero_rate=0.01
    )
    silence_total = 0
    for silence in silences:
        li = silence[1] - silence[0]
        silence_total += li
    silence_rate = silence_total / l

    # 非静音段方差
    last_e = 0
    non_silences = list()
    for silence in silences:
        b, e = silence
        if b > last_e:
            non_silences.append([last_e, b])
        last_e = e
    else:
        if l > last_e:
            non_silences.append([last_e, l])

    # 静音段的数量
    silence_count = len(non_silences)

    if silence_count == 0:
        mean_non_silence = 0
        var_non_silence = 0
        var_var_non_silence = 0
        var_non_silence_rate = 1
    else:
        signal_non_silences = list()
        for non_silence in non_silences:
            b, e = non_silence
            signal_non_silences.append(signal[b: e])

        # 非静音段, 各段方差的方差.
        v = list()
        for signal_non_silence in signal_non_silences:
            v.append(np.var(signal_non_silence))
        var_var_non_silence = np.var(v)

        signal_non_silences = np.concatenate(signal_non_silences)
        # 非静音段整体均值
        mean_non_silence = np.mean(signal_non_silences)
        # 非静音段整体方差
        var_non_silence = np.var(signal_non_silences)
        # 非静音段整体方差 除以 整体方差
        var_non_silence_rate = var_non_silence / var

    # 全段, 分段方差的方差
    sub_signal_list = np.split(signal, 20)

    whole_var = list()
    for sub_signal in sub_signal_list:
        sub_var = np.var(sub_signal)
        whole_var.append(sub_var)
    var_var_whole = np.var(whole_var)

    result = {
        'mean': float(mean),
        'var': float(var),
        'per1': float(per1),
        'per25': float(per25),
        'per50': float(per50),
        'per75': float(per75),
        'per99': float(per99),
        'silence_rate': float(silence_rate),
        'mean_non_silence': float(mean_non_silence),
        'silence_count': float(silence_count),
        'var_var_non_silence': float(var_var_non_silence),
        'var_non_silence': float(var_non_silence),
        'var_non_silence_rate': float(var_non_silence_rate),
        'var_var_whole': float(var_var_whole),

    }
    return result


if __name__ == '__main__':
    pass
