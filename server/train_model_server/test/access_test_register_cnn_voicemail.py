#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json

import requests


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--host',
        # default='127.0.0.1',
        default='10.75.27.247',
        type=str
    )
    parser.add_argument('--port', default=9180, type=int)
    parser.add_argument(
        '--language',
        default='en-US',
        # default='id-ID',
        type=str
    )
    parser.add_argument('--increase_number', default=5000, type=int)

    args = parser.parse_args()
    return args


def main():
    # curl -X POST http://127.0.0.1:4070/HeartBeat -d '{"valStr": "tianxing", "valInt": 20221008}'
    args = get_args()

    url = 'http://{host}:{port}/register/cnn_voicemail'.format(
        host=args.host,
        port=args.port,
    )

    data = {
        'language': args.language,
        'increase_number': args.increase_number,
        # 'data_dir': r'D:\programmer\asr_datasets\voicemail\origin_wav\zh-TW\wav_segmented'
    }

    headers = {
        'Content-Type': 'application/json'
    }

    resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=None)
    print(resp.text)
    return


if __name__ == '__main__':
    main()
