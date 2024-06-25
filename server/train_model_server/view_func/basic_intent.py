#!/usr/bin/python3
# -*- coding: utf-8 -*-
import base64
import io
import logging

from flask import request
import json
import jsonschema
import numpy as np
from scipy.io import wavfile

from server.exception import ExpectedError
from server.flask_server.route_wrap.common_route_wrap import common_route_wrap
from toolbox.logging.misc import json_2_str

from server.train_model_server.schema.basic_intent import basic_intent_by_language_schema, \
    basic_intent_by_language_pivot_table_schema
from server.train_model_server.service.basic_intent import get_basic_intent_by_language_service_instance


logger = logging.getLogger('server')


@common_route_wrap
def basic_intent_by_language_view_func():
    args = request.json
    logger.info('args: {}'.format(json_2_str(args)))

    # 请求体校验
    try:
        jsonschema.validate(args, basic_intent_by_language_schema)
    except (jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError, ) as e:
        raise ExpectedError(
            status_code=60401,
            message='request body invalid. ',
            detail=str(e)
        )

    language = args['language']
    text = args['text']

    service = get_basic_intent_by_language_service_instance()
    result = service.forward(text, language)

    return result


@common_route_wrap
def basic_intent_by_language_pivot_table_view_func():
    args = request.json
    logger.info('args: {}'.format(json_2_str(args)))

    # 请求体校验
    try:
        jsonschema.validate(args, basic_intent_by_language_pivot_table_schema)
    except (jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError, ) as e:
        raise ExpectedError(
            status_code=60401,
            message='request body invalid. ',
            detail=str(e)
        )

    language = args['language']

    service = get_basic_intent_by_language_service_instance()
    result = service.get_pivot_table(language)

    return result


if __name__ == '__main__':
    pass
