#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import pandas as pd
from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--intent_classification_xlsx',
        # default='datasets/basic_intent_classification/intent_classification_cn.xlsx',
        default=project_path / "datasets/waba_intent_classification.xlsx",
        type=str
    )
    parser.add_argument('--train_labeled', default='train_labeled.json', type=str)
    parser.add_argument('--valid_labeled', default='valid_labeled.json', type=str)
    parser.add_argument('--train_all', default='train_all.json', type=str)

    parser.add_argument('--dataset_excel', default='dataset.xlsx', type=str)

    args = parser.parse_args()
    return args


def make_dataset(args):
    df = pd.read_excel(args.intent_classification_xlsx)

    dataset = list()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        label = row['label1']
        selected = row['selected']
        random1 = random.random()
        random2 = random.random()

        if pd.isna(text):
            continue

        if pd.isna(selected):
            selected = None
        else:
            selected = int(selected)

        dataset.append({
            'text': text,
            'label': label if selected == 1 else '无关领域',
            'selected': selected,
            'random1': random1,
            'random2': random2,

        })

    dataset = pd.DataFrame(dataset)
    dataset = dataset.sort_values(by='random1', ascending=False)
    dataset.to_excel(args.dataset_excel, index=False, encoding='utf_8_sig')
    return


def main():
    args = get_args()

    make_dataset(args)
    df = pd.read_excel(args.dataset_excel)

    train_labeled_f = open(args.train_labeled, 'w', encoding='utf-8')
    valid_labeled_f = open(args.valid_labeled, 'w', encoding='utf-8')
    train_all_f = open(args.train_all, 'w', encoding='utf-8')

    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        label = row['label']
        selected = row['selected']
        random2 = row['random2']

        label = label if selected == 1 else '无关领域'

        d = {
            'text': text,
            'label': label,
            'selected': selected,
            'random2': random2,
        }
        d = json.dumps(d, ensure_ascii=False)

        train_all_f.write('{}\n'.format(d))

        if selected == 1 and label != '无关领域':
            if random2 < 0.8:
                train_labeled_f.write('{}\n'.format(d))
            else:
                valid_labeled_f.write('{}\n'.format(d))

    train_labeled_f.close()
    valid_labeled_f.close()
    train_all_f.close()

    return


if __name__ == '__main__':
    main()
