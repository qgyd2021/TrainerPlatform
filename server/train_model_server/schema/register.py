#!/usr/bin/python3
# -*- coding: utf-8 -*-


register_cnn_voicemail_schema = {
    'type': 'object',
    'required': ['language', 'increase_number'],
    'properties': {
        'language': {
            'type': 'string'
        },
        'increase_number': {
            'type': 'integer'
        },
        'data_dir': {
            'type': 'string'
        }
    }
}


if __name__ == '__main__':
    pass
