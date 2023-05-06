#!/usr/bin/python3
# -*- coding: utf-8 -*-


cnn_voicemail_schema = {
    'type': 'object',
    'required': ['language', 'signal'],
    'properties': {
        'language': {
            'type': 'string',
        },
        'signal': {
            'type': 'string',
        },
    }
}


cnn_voicemail_pivot_table_schema = {
    'type': 'object',
    'required': ['language'],
    'properties': {
        'language': {
            'type': 'string',
        },
    }
}

if __name__ == '__main__':
    pass
