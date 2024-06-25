#!/usr/bin/python3
# -*- coding: utf-8 -*-


forward_request_schema = {
    'type': 'object',
    'required': ['model_path', 'inputs'],
    'properties': {
        'model_path': {
            'type': 'string',
        },
        'inputs': {
            'type': 'object',
        },
    }
}


if __name__ == '__main__':
    pass
