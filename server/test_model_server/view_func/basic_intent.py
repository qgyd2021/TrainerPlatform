#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from glob import glob

from flask import render_template, request
import jsonschema

from project_settings import project_path
from server.exception import ExpectedError
from server.flask_server.route_wrap.common_route_wrap import common_route_wrap
from toolbox.logging.misc import json_2_str
from server.test_model_server import settings

logger = logging.getLogger('server')


def test_model_view_func():
    return render_template('basic_intent.html')


@common_route_wrap
def get_available_models_view_func():
    args = request.form
    logger.info('args: {}'.format(json_2_str(args)))

    models_dir = Path(settings.models_dir)
    available_models = models_dir.glob('basic_intent_*')
    available_models = [m.name for m in available_models]
    return available_models


@common_route_wrap
def forward_view_func():
    args = request.form
    logger.info('args: {}'.format(json_2_str(args)))


    models_dir = Path(settings.models_dir)
    available_models = models_dir.glob('basic_intent_*')
    available_models = [m.name for m in available_models]
    return available_models


if __name__ == '__main__':
    pass
