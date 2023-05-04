#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from glob import glob

from flask import render_template, request
import jsonschema
import torch

from project_settings import project_path
from server.exception import ExpectedError
from server.flask_server.route_wrap.common_route_wrap import common_route_wrap
from toolbox.logging.misc import json_2_str
from server.train_model_server.schema.forward import forward_request_schema


logger = logging.getLogger('server')


@common_route_wrap
def forward_view_func():
    args = request.json
    logger.info('args: {}'.format(json_2_str(args)))

    # 请求体校验
    try:
        jsonschema.validate(args, forward_request_schema)
    except (jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError, ) as e:
        raise ExpectedError(
            status_code=60401,
            message='request body invalid. ',
            detail=str(e)
        )

    model_path = args['model_path']
    inputs = args['inputs']

    trace_model = torch.jit.load(model_path)

    return []


if __name__ == '__main__':
    pass
