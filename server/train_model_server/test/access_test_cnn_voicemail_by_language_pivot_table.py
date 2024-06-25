#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import base64
import hashlib
import json
import os
import time

import requests

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--host', default='127.0.0.1', type=str)
    parser.add_argument('--host', default='10.75.27.247', type=str)
    parser.add_argument('--port', default=9180, type=int)
    parser.add_argument(
        '--filename',
        default='00b08df0-c8c7-4905-9830-0bb333f2dfe3_en-US_1666032716.6763692.wav',
        type=str
    )
    parser.add_argument('--language', default='en-US', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    url = 'http://{host}:{port}/cnn_voicemail_by_language_pivot_table'.format(
        host=args.host,
        port=args.port,
    )

    headers = {
        'Content-Type': 'application/json'
    }

    # language = args.language
    language = 'zh-TW'

    data = {
        'language': language,
        'call_id': 'access_test_call_id_cnn_voicemail_pivot_table',
        'scene_id': 'access_test_scene_id_cnn_voicemail_pivot_table',
        'verbose': True,
    }

    resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=2)
    print(resp.status_code)
    print(resp.text)

    return


if __name__ == '__main__':
    main()
