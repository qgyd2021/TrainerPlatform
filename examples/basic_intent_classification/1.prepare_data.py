#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import random
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import pandas as pd
from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--without_irrelevant_domain', action='store_true')
    parser.add_argument('--dataset_filename', default='dataset.xlsx', type=str)
    parser.add_argument('--do_lowercase', action='store_true')

    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='valid.json', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    n_hierarchical = 2

    df = pd.read_excel(args.dataset_filename)
    df = df[df['selected'] == 1]

    dataset = list()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        label0 = row['label0']
        if args.without_irrelevant_domain and label0 == '无关领域':
            continue

        text = str(text)
        if args.do_lowercase:
            text = text.lower()

        labels = {'label{}'.format(idx): str(row['label{}'.format(idx)]) for idx in range(n_hierarchical)}

        random1 = random.random()
        random2 = random.random()

        dataset.append({
            'text': text,
            **labels,

            'random1': random1,
            'random2': random2,
            'flag': 'TRAIN' if random2 < 0.8 else 'TEST',
        })

    dataset = list(sorted(dataset, key=lambda x: x['random1'], reverse=True))

    f_train = open(args.train_subset, 'w', encoding='utf-8')
    f_test = open(args.valid_subset, 'w', encoding='utf-8')

    for row in tqdm(dataset):

        flag = row['flag']
        row = json.dumps(row, ensure_ascii=False)
        if flag == 'TRAIN':
            f_train.write('{}\n'.format(row))
        else:
            f_test.write('{}\n'.format(row))

    f_train.close()
    f_test.close()
    return


if __name__ == '__main__':
    main()
