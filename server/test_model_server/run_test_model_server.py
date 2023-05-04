#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
from datetime import datetime

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

from flask import Flask
from gevent import pywsgi

from server import log
from server.test_model_server import settings

log.setup(log_directory=settings.log_directory)

from server.flask_server.view_func.heart_beat import heart_beat
from server.test_model_server.view_func.basic_intent import test_model_view_func, get_available_models_view_func
from server.test_model_server.view_func.forward import forward_view_func


logger = logging.getLogger('server')


# 初始化服务
flask_app = Flask(
    __name__,
    static_url_path='/',
    static_folder='static',
    template_folder='static/templates',
)

flask_app.add_url_rule(rule='/HeartBeat', view_func=heart_beat, methods=['GET', 'POST'], endpoint='HeartBeat')
flask_app.add_url_rule(rule='/basic_intent', view_func=test_model_view_func, methods=['GET', 'POST'], endpoint='BasicIntent')
flask_app.add_url_rule(rule='/basic_intent/get_available_models', view_func=get_available_models_view_func, methods=['GET', 'POST'], endpoint='GetAvailableModels')
flask_app.add_url_rule(rule='/forward', view_func=forward_view_func, methods=['GET', 'POST'], endpoint='Forward')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        default=settings.port,
        type=int,
    )
    args = parser.parse_args()

    logger.info('model server is already, 127.0.0.1:{}'.format(args.port))

    # flask_app.run(
    #     host='0.0.0.0',
    #     port=args.port,
    # )

    server = pywsgi.WSGIServer(
        listener=('0.0.0.0', args.port),
        application=flask_app
    )
    server.serve_forever()
