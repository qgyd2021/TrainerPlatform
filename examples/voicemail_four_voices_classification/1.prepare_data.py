#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename_patterns', nargs='+')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    filename_patterns = args.filename_patterns
    print(filename_patterns)
    return


if __name__ == '__main__':
    main()
