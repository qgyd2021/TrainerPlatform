#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://github.com/igormq/ctcdecode-pytorch
"""
from collections import defaultdict
import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np


class PrefixBeamSearch(object):
    """
    https://arxiv.org/abs/1408.2873
    https://blog.csdn.net/weixin_42615068/article/details/93767781
    """
    def __init__(self, beam_size=100, blank=0, cut_off_prob=1.0, top_n=None):
        """
        :param beam_size: 束大小.每个时间步将保留 beam_size 个概率最大的候选项.
        :param blank: black 标签对应的索引.
        :param cut_off_prob: 每个时间步, 与当前最大概率的比值小于 cut_off_prob 的, 将不参与搜索.
        :param top_n: 每个时间步, 最多只有 top_n 个节点参与搜索.
        """
        self.beam_size = beam_size
        self.blank = blank
        self.cut_off_prob = cut_off_prob
        self.top_n = top_n

    @staticmethod
    def max_rate(x):
        m = np.max(x)
        y = x - m
        y = np.exp(y)
        return y

    def search(self, log_probs_sequence: np.ndarray):
        """
        quick prefix beam search
        :param log_probs_sequence: np.ndarray, shape=[time_steps, alphabet_size]
        :return:
        """
        time_steps, alphabet_size = log_probs_sequence.shape
        top_n = min(self.top_n, alphabet_size) if self.top_n else alphabet_size

        prev_beams = {
            (): {
                'end_with_blank_log_prob': 0.0,
                'end_with_non_blank_log_prob': -float('inf'),
            }
        }
        prev_beams = list(sorted(
            prev_beams.items(),
            key=lambda x: np.logaddexp(
                x[1]['end_with_blank_log_prob'],
                x[1]['end_with_non_blank_log_prob']
            ),
            reverse=True,
        ))

        for time_step in range(time_steps):
            log_probs = log_probs_sequence[time_step]

            next_beams: Dict[tuple, dict] = defaultdict(lambda: {
                'end_with_blank_log_prob': -float('inf'),
                'end_with_non_blank_log_prob': -float('inf'),
            })

            if self.cut_off_prob < 1.0 or top_n < alphabet_size:
                idxs = np.argsort(log_probs)[::-1]
                log_probs_sorted = log_probs[idxs]

                n_idxs = min((self.max_rate(log_probs_sorted) <= self.cut_off_prob).sum(), top_n, alphabet_size)

                pruned_indexes = idxs[:n_idxs].tolist()
            else:
                pruned_indexes = np.argsort(log_probs)[::-1]

            for index in pruned_indexes:
                if len(next_beams) > self.beam_size:
                    break
                for prefix, prev_beam in prev_beams:
                    if len(next_beams) > self.beam_size:
                        break

                    log_prob = log_probs[index]
                    p_b = prev_beam['end_with_blank_log_prob']
                    p_nb = prev_beam['end_with_non_blank_log_prob']

                    if index == self.blank:
                        existing_beam = next_beams[prefix]
                        next_beam = {
                            'end_with_blank_log_prob': np.logaddexp(
                                existing_beam['end_with_blank_log_prob'],
                                log_prob + np.logaddexp(p_b, p_nb)
                            ),
                            'end_with_non_blank_log_prob': existing_beam['end_with_non_blank_log_prob']
                        }
                        next_beams[prefix] = next_beam
                    elif len(prefix) > 0 and index == prefix[-1]:
                        existing_beam = next_beams[prefix]
                        next_beam = {
                            'end_with_blank_log_prob': existing_beam['end_with_blank_log_prob'],
                            'end_with_non_blank_log_prob': np.logaddexp(
                                existing_beam['end_with_non_blank_log_prob'],
                                log_prob + p_nb
                            ),
                        }
                        next_beams[prefix] = next_beam

                        prefix_extend = prefix + (index,)
                        existing_beam = next_beams[prefix_extend]
                        if existing_beam['end_with_non_blank_log_prob'] > -float('inf'):
                            next_beam = {
                                'end_with_blank_log_prob': existing_beam['end_with_blank_log_prob'],
                                'end_with_non_blank_log_prob': np.logaddexp(
                                    existing_beam['end_with_non_blank_log_prob'],
                                    log_prob + p_b
                                ),
                            }
                            next_beams[prefix_extend] = next_beam
                    else:
                        prefix_extend = prefix + (index,)
                        existing_beam = next_beams[prefix_extend]
                        next_beam = {
                            'end_with_blank_log_prob': existing_beam['end_with_blank_log_prob'],
                            'end_with_non_blank_log_prob': np.logaddexp(
                                existing_beam['end_with_non_blank_log_prob'],
                                log_prob + np.logaddexp(p_b, p_nb)
                            ),
                        }
                        next_beams[prefix_extend] = next_beam

            next_beams = next_beams.items()
            next_beams = list(sorted(
                next_beams,
                key=lambda x: np.logaddexp(x[1]['end_with_blank_log_prob'], x[1]['end_with_non_blank_log_prob']),
                reverse=True,
            ))
            next_beams = next_beams[:self.beam_size]
            prev_beams = next_beams

        result = [
            (
                np.logaddexp(
                    prev_beam['end_with_blank_log_prob'],
                    prev_beam['end_with_non_blank_log_prob']
                ),
                prefix,
            )
            for prefix, prev_beam in prev_beams
        ]
        return result


def demo1():
    vocab_list = ["<blank>", "\'", ' ', 'a', 'b', 'c', 'd']

    log_probs_seq1 = np.array([[0.1649, 0.0639, 0.2112, 0.2732, 0.0687, 0.0361, 0.1818],
                               [0.0689, 0.0331, 0.2287, 0.2439, 0.0970, 0.3190, 0.0095],
                               [0.0812, 0.2181, 0.1999, 0.1825, 0.0850, 0.1490, 0.0842],
                               [0.0977, 0.1209, 0.1916, 0.0147, 0.2805, 0.2425, 0.0521],
                               [0.0195, 0.1333, 0.0055, 0.0030, 0.2175, 0.2080, 0.4132],
                               [0.0146, 0.1647, 0.1981, 0.1907, 0.1896, 0.1986, 0.0438]])
    log_probs_seq1 = np.log(log_probs_seq1)

    prefix_beam_search = PrefixBeamSearch(beam_size=20, blank=0, cut_off_prob=0.9)

    beam_result = prefix_beam_search.search(log_probs_seq1)
    print(beam_result[0])
    for score, path in beam_result:

        string = [vocab_list[s] for s in path]
        string = ''.join(string)
        print(score)
        print(string)
    return


if __name__ == '__main__':
    demo1()
