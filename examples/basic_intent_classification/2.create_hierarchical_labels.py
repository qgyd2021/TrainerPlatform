#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import OrderedDict
import os
from pathlib import Path
import pickle
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./', type=str)
    parser.add_argument('--dataset_filename', default='dataset.xlsx', type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    dataset_filename = args.dataset_filename

    file_dir = Path(args.file_dir)
    file_dir.mkdir(exist_ok=True)

    n_hierarchical = 2

    df = pd.read_excel(file_dir / dataset_filename)
    df = df[df['selected'] == 1]

    # 生成 hierarchical_labels
    temp_hierarchical_labels = OrderedDict()

    for i, row in df.iterrows():
        text = row['text']
        label0 = row['label0']
        label1 = row['label1']

        if temp_hierarchical_labels.get(label0) is None:
            temp_hierarchical_labels[label0] = list()

        if label1 not in temp_hierarchical_labels[label0]:
            temp_hierarchical_labels[label0].append(label1)

    if n_hierarchical > 2:
        hierarchical_labels = OrderedDict()
        for idx in range(n_hierarchical - 2):
            for k, v in temp_hierarchical_labels.items():
                parent, label = k.rsplit('_', maxsplit=1)

                if hierarchical_labels.get(parent) is None:
                    hierarchical_labels[parent] = OrderedDict({
                        label: v
                    })
                else:
                    if hierarchical_labels[parent].get(label) is None:
                        hierarchical_labels[parent][label] = v
    else:
        hierarchical_labels = temp_hierarchical_labels

    with open(file_dir / 'hierarchical_labels.pkl', 'wb') as f:
        pickle.dump(hierarchical_labels, f)

    with open(file_dir / 'hierarchical_labels.pkl', 'rb') as f:
        hierarchical_labels = pickle.load(f)

    print(hierarchical_labels)
    return


if __name__ == '__main__':
    main()
