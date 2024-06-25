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
from server.train_model_server.service.cnn_voicemail import get_cnn_voicemail_by_language_service_instance, \
    get_cnn_voicemail_common_service_instance
from server.train_model_server.schema.cnn_voicemail import cnn_voicemail_by_language_schema, \
    cnn_voicemail_by_language_pivot_table_schema, cnn_voicemail_common_schema, \
    cnn_voicemail_schema, cnn_voicemail_correction_schema
from toolbox.logging.misc import json_2_str

logger = logging.getLogger('server')


@common_route_wrap
def cnn_voicemail_by_language_view_func():
    args = request.json
    logger.info('args: {}'.format(json_2_str(args)))

    # 请求体校验
    try:
        jsonschema.validate(args, cnn_voicemail_by_language_schema)
    except (jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError, ) as e:
        raise ExpectedError(
            status_code=60401,
            message='request body invalid. ',
            detail=str(e)
        )

    language = args['language']
    base64string = args['signal']

    # wav 音频解码.
    try:
        base64byte = base64string.encode('utf-8')
        wav_bytes = base64.b64decode(base64byte)

        f = io.BytesIO(wav_bytes)
        sample_rate, signal = wavfile.read(f)
    except Exception as e:
        raise ExpectedError(
            status_code=60401,
            message="base64 decode failed. ",
            detail="error: {}; tips: [base64string = base64.b64encode(data).decode('utf-8')]. ".format(str(e)),
        )

    service = get_cnn_voicemail_by_language_service_instance()

    predicts = service.forward(signal, language)

    return predicts


@common_route_wrap
def cnn_voicemail_by_language_pivot_table_view_func():
    args = request.json
    logger.info('args: {}'.format(json_2_str(args)))

    # 请求体校验
    try:
        jsonschema.validate(args, cnn_voicemail_by_language_pivot_table_schema)
    except (jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError, ) as e:
        raise ExpectedError(
            status_code=60401,
            message='request body invalid. ',
            detail=str(e)
        )

    language = args['language']

    service = get_cnn_voicemail_by_language_service_instance()

    pivot_table = service.get_pivot_table(language)
    print(pivot_table)
    return pivot_table


@common_route_wrap
def cnn_voicemail_common_view_func():
    args = request.json
    logger.info('args: {}'.format(json_2_str(args)))

    # 请求体校验
    try:
        jsonschema.validate(args, cnn_voicemail_common_schema)
    except (jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError, ) as e:
        raise ExpectedError(
            status_code=60401,
            message='request body invalid. ',
            detail=str(e)
        )

    base64string = args['signal']

    # wav 音频解码.
    try:
        base64byte = base64string.encode('utf-8')
        wav_bytes = base64.b64decode(base64byte)

        f = io.BytesIO(wav_bytes)
        sample_rate, signal = wavfile.read(f)
    except Exception as e:
        raise ExpectedError(
            status_code=60401,
            message="base64 decode failed. ",
            detail="error: {}; tips: [base64string = base64.b64encode(data).decode('utf-8')]. ".format(str(e)),
        )

    service = get_cnn_voicemail_common_service_instance()

    predicts = service.forward(signal)

    return predicts


@common_route_wrap
def cnn_voicemail_view_func():
    args = request.json
    logger.info('args: {}'.format(json_2_str(args)))

    # 请求体校验
    try:
        jsonschema.validate(args, cnn_voicemail_schema)
    except (jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError, ) as e:
        raise ExpectedError(
            status_code=60401,
            message='request body invalid. ',
            detail=str(e)
        )

    language = args['language']
    base64string = args['signal']

    # wav 音频解码.
    try:
        base64byte = base64string.encode('utf-8')
        wav_bytes = base64.b64decode(base64byte)

        f = io.BytesIO(wav_bytes)
        sample_rate, signal = wavfile.read(f)
    except Exception as e:
        raise ExpectedError(
            status_code=60401,
            message="base64 decode failed. ",
            detail="error: {}; tips: [base64string = base64.b64encode(data).decode('utf-8')]. ".format(str(e)),
        )

    result = list()

    service1 = get_cnn_voicemail_by_language_service_instance()
    try:
        predicts1 = service1.forward(signal, language)
    except ExpectedError:
        predicts1 = {'prob': 1.0, 'label': 'non_voicemail'}
    if predicts1['label'] == 'voicemail':
        result.append(predicts1)
        return result

    service2 = get_cnn_voicemail_common_service_instance()
    predicts2 = service2.forward(signal)
    predicts2['prob'] = round(predicts2['prob'] * predicts1['prob'], 4)
    result.append(predicts2)
    return result


@common_route_wrap
def cnn_voicemail_correction_view_func():
    args = request.json
    logger.info('args: {}'.format(json_2_str(args)))

    # 请求体校验
    try:
        jsonschema.validate(args, cnn_voicemail_correction_schema)
    except (jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError, ) as e:
        raise ExpectedError(
            status_code=60401,
            message='request body invalid. ',
            detail=str(e)
        )

    language = args['language']

    service = get_cnn_voicemail_by_language_service_instance()
    correction = service.get_correction(language)

    return correction


if __name__ == '__main__':
    pass
