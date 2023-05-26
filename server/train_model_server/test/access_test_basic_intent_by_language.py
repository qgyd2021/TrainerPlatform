#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import base64
import hashlib
import json
import os
import platform
import time

import requests

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1', type=str)
    # parser.add_argument('--host', default='10.75.27.247', type=str)
    parser.add_argument('--port', default=9180, type=int)
    parser.add_argument(
        '--text',
        default='who are you may i know',
        type=str
    )
    parser.add_argument('--language', default='en', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    url = 'http://{host}:{port}/basic_intent_by_language'.format(
        host=args.host,
        port=args.port,
    )

    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        'text': args.text,
        'language': args.language,
    }

    resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=2)
    print(resp.status_code)

    if platform.system() == 'Windows':
        print(resp.json())
    else:
        print(resp.text)
    return


if __name__ == '__main__':
    main()
