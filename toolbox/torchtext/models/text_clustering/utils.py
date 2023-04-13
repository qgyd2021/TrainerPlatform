#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score


def clustering_accuracy(y_pred, y_true):
    """
    :param y_pred: np.ndarray, shape=[n,], dtype np.int64
    :param y_true: np.ndarray, shape=[n,], dtype np.int64
    """
    max_pred = np.max(y_pred)
    max_true = np.max(y_true)
    size = max(max_pred, max_true) + 1

    weight = np.zeros(shape=(size, size), dtype=np.int64)

    l = y_pred.shape[0]
    for i in range(l):
        weight[y_pred[i], y_true[i]] += 1

    index: np.ndarray = linear_assignment(- weight)

    accuracy = sum([weight[i, j] for i, j in index]) * 1.0 / l

    return accuracy


def clustering_score(y_true, y_pred):
    """
    :param y_pred: np.ndarray, shape=[n,], dtype np.int64
    :param y_true: np.ndarray, shape=[n,], dtype np.int64
    """
    acc = clustering_accuracy(y_pred, y_true)

    ari = adjusted_rand_score(y_true, y_pred)

    mni = normalized_mutual_info_score(
        labels_true=y_true,
        labels_pred=y_pred,
        average_method='geometric'
    )
    result = {
        'ACC': round(acc, 4),
        'ARI': round(ari, 4),
        'NMI': round(mni, 4)

    }
    return result


def demo1():
    import numpy as np

    y_pred = np.array([0, 2, 2, 1, 0, 1, 2, 1, 0, 1, 0], dtype=np.int64)
    y_true = np.array([0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.int64)

    accuracy = clustering_accuracy(y_pred, y_true)
    print(accuracy)

    mni = normalized_mutual_info_score(
        labels_true=y_true,
        labels_pred=y_pred,
        average_method='geometric'
    )
    print(mni)

    ari = adjusted_rand_score(y_true, y_pred)
    print(ari)

    scores = clustering_score(y_true, y_pred)
    print(scores)
    return


if __name__ == '__main__':
    demo1()
