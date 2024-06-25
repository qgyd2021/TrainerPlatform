#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from glob import glob
import json
import os
from pathlib import Path
import random
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./', type=str)
    parser.add_argument('--filename_patterns', nargs='+')

    args = parser.parse_args()
    return args


def get_dataset(args):
    file_dir = Path(args.file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    filename_patterns = args.filename_patterns
    print('filename_patterns: {}'.format(filename_patterns))

    # white_noise, noise, voice
    folder_to_label = {
        'bell': 'bell',
        'white_noise': 'white_noise',
        'low_white_noise': 'white_noise',
        'high_white_noise': 'noise',
        'music': 'music',
        'mute': 'mute',
        'noise': 'noise',
        'noise_mute': 'noise_mute',
        # 'special': 'voicemail',
        'voice': 'voice',
        'voicemail': 'voicemail',
        # 'non_voicemail': 'non_voicemail',
    }

    result = list()
    for filename_pattern in filename_patterns:
        filename_list = glob(filename_pattern)
        for filename in tqdm(filename_list):
            sample_rate, signal = wavfile.read(filename)
            if len(signal) < sample_rate * 2:
                continue

            path, fn = os.path.split(filename)
            basename, ext = os.path.splitext(fn)
            path, folder = os.path.split(path)
            path, _ = os.path.split(path)
            path, language = os.path.split(path)

            if folder not in folder_to_label.keys():
                continue

            label = folder_to_label[folder]
            random1 = random.random()
            random2 = random.random()

            result.append({
                'filename': filename,
                'folder': folder,
                'label': label,
                'language': language,
                'random1': random1,
                'random2': random2,
                'flag': 'TRAIN' if random2 < 0.8 else 'TEST',
            })

    df = pd.DataFrame(result)
    pivot_table = pd.pivot_table(df, index=['label'], values=['filename'], aggfunc='count')
    print(pivot_table)

    df = df.sort_values(by=['random1'], ascending=False)
    df.to_excel(file_dir / 'dataset.xlsx', index=False, encoding='utf_8_sig')
    return


def split_dataset(args):
    file_dir = Path(args.file_dir)
    file_dir.mkdir(exist_ok=True)

    df = pd.read_excel(file_dir / 'dataset.xlsx')

    train = list()
    test = list()

    for i, row in df.iterrows():
        flag = row['flag']
        if flag == 'TRAIN':
            train.append(row)
        else:
            test.append(row)

    train = pd.DataFrame(train)
    train.to_excel(file_dir / 'train.xlsx', index=False, encoding='utf_8_sig')
    test = pd.DataFrame(test)
    test.to_excel(file_dir / 'test.xlsx', index=False, encoding='utf_8_sig')

    return


def main():
    args = get_args()
    get_dataset(args)
    split_dataset(args)
    return


if __name__ == '__main__':
    main()
