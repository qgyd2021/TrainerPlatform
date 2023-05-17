#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import random

import pandas as pd


def demo0():
    filename = r'D:\Users\tianx\PycharmProjects\PyTorch\练习实例\文本分类\层级文本分类\意图分类\意图分类 - 汉语.xlsx'

    df = pd.read_excel(filename)

    dataset = list()
    for i, row in df.iterrows():
        text = row['text']
        label = row['label1']
        selected = row['selected']
        random1 = random.random()
        random2 = random.random()

        if pd.isna(text):
            continue

        dataset.append({
            'text': text,
            'label': label if selected == 1 else '无关领域',
            'selected': selected,
            'random1': random1,
            'random2': random2,

        })

    dataset = pd.DataFrame(dataset)
    dataset = dataset.sort_values(by='random1', ascending=False)
    dataset.to_excel('dataset.xlsx', index=False, encoding='utf_8_sig')
    return


def demo1():
    df = pd.read_excel('dataset.xlsx')

    train_labeled_f = open('train_labeled.json', 'w', encoding='utf-8')
    valid_labeled_f = open('valid_labeled.json', 'w', encoding='utf-8')
    train_all_f = open('train_all.json', 'w', encoding='utf-8')

    for i, row in df.iterrows():
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
    # demo0()
    demo1()
