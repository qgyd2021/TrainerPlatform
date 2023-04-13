#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://github.com/igormq/ctcdecode-pytorch
"""
from typing import List
import torch
import numpy as np


class GreedyDecoder(object):

    def search(self, log_probs_seq, blank=0):
        max_indexes = np.argmax(log_probs_seq, axis=1)
        max_log_probs = np.max(log_probs_seq, axis=1)

        mask = np.concatenate([
            np.array([1]),
            np.abs(max_indexes[:-1] - max_indexes[1:]) > 0
        ])
        mask = np.array(mask, dtype=np.bool)
        mask = mask * (max_indexes != blank)

        path: np.ndarray = max_indexes[mask]
        indexes: List[int] = path.tolist()
        score = np.sum(max_log_probs[mask])

        return float(score), indexes


def demo1():
    vocab_list = ["<blank>", "\'", ' ', 'a', 'b', 'c', 'd']

    log_probs_seq1 = np.array([[0.1649, 0.0639, 0.2112, 0.2732, 0.0687, 0.0361, 0.1818],
                               [0.0689, 0.0331, 0.2287, 0.2439, 0.0970, 0.3190, 0.0095],
                               [0.0812, 0.2181, 0.1999, 0.1825, 0.0850, 0.1490, 0.0842],
                               [0.0977, 0.1209, 0.1916, 0.0147, 0.2805, 0.2425, 0.0521],
                               [0.0195, 0.1333, 0.0055, 0.0030, 0.2175, 0.2080, 0.4132],
                               [0.0146, 0.1647, 0.1981, 0.1907, 0.1896, 0.1986, 0.0438]])

    log_probs_seq1 = np.log(log_probs_seq1)

    greedy_result = ["ac'bdc", "b'da"]

    ctc_greedy_decoder = GreedyDecoder()

    score, indexes = ctc_greedy_decoder.search(log_probs_seq1)

    best_result = [vocab_list[i] for i in indexes]
    best_result = ''.join(best_result)
    print(best_result)
    return


if __name__ == '__main__':
    demo1()
