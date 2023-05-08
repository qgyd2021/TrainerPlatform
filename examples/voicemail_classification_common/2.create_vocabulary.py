#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./', type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    file_dir = Path(args.file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = Vocabulary()

    vocabulary.add_token_to_namespace('white_noise', 'labels')
    vocabulary.add_token_to_namespace('voicemail', 'labels')
    vocabulary.add_token_to_namespace('voice', 'labels')
    vocabulary.add_token_to_namespace('noise', 'labels')
    vocabulary.add_token_to_namespace('bell', 'labels')
    vocabulary.add_token_to_namespace('mute', 'labels')
    vocabulary.add_token_to_namespace('noise_mute', 'labels')
    vocabulary.add_token_to_namespace('music', 'labels')

    vocabulary.save_to_files(file_dir / 'vocabulary')
    return


if __name__ == '__main__':
    main()
