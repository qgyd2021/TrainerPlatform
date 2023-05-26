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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./', type=str)

    parser.add_argument('--task', default='default', type=str)
    parser.add_argument('--filename_patterns', nargs='+')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pass
