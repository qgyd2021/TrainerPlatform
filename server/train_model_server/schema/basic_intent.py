#!/usr/bin/python3
# -*- coding: utf-8 -*-


basic_intent_by_language_schema = {
    'type': 'object',
    'required': ['language', 'text'],
    'properties': {
        'language': {
            'type': 'string',
        },
        'text': {
            'type': 'string',
        },
    }
}


basic_intent_by_language_pivot_table_schema = {
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
