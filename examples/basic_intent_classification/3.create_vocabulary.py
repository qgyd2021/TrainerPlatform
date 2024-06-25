#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import pickle
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import pandas as pd

from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_dir', required=True, type=str)
    parser.add_argument('--hierarchical_labels_pkl', default='hierarchical_labels.pkl', type=str)
    parser.add_argument('--vocabulary', default='vocabulary', type=str)
    parser.add_argument('--labels_json', default='labels.json', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.hierarchical_labels_pkl, 'rb') as f:
        hierarchical_labels = pickle.load(f)
    print(hierarchical_labels)
    # 深度遍历
    token_to_index = OrderedDict()
    tasks = [hierarchical_labels]
    while len(tasks) != 0:
        task = tasks.pop(0)
        for parent, downstream in task.items():
            if isinstance(downstream, list):
                for label in downstream:
                    if pd.isna(label):
                        continue
                    label = '{}_{}'.format(parent, label)
                    token_to_index[label] = len(token_to_index)
            elif isinstance(downstream, OrderedDict):
                new_task = OrderedDict()
                for k, v in downstream.items():
                    new_task['{}_{}'.format(parent, k)] = v
                tasks.append(new_task)
            else:
                raise NotImplementedError

    vocabulary = Vocabulary(non_padded_namespaces=['tokens', 'labels'])
    for label, index in token_to_index.items():
        vocabulary.add_token_to_namespace(label, namespace='labels')

    vocabulary.set_from_file(
        filename=os.path.join(args.pretrained_model_dir, 'vocab.txt'),
        is_padded=False,
        oov_token='[UNK]',
        namespace='tokens',
    )
    vocabulary.save_to_files(args.vocabulary)

    # labels.json
    token_to_index = vocabulary.get_token_to_index_vocabulary(namespace='labels')
    labels = list()
    for k, v in sorted(token_to_index.items(), key=lambda x: x[1]):
        labels.append(k)
    with open(args.labels_json, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)

    print('注意检查 Vocabulary 中标签的顺序与 hierarchical_labels 是否一致. ')
    return


if __name__ == '__main__':
    main()
