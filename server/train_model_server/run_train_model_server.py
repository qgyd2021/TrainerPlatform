#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

from flask import Flask
from gevent import pywsgi

from server import log
from server.train_model_server import settings

log.setup(log_directory=settings.log_directory)

from server.flask_server.view_func.heart_beat import heart_beat
from server.train_model_server.view_func.register import register_cnn_voicemail_view_func

logger = logging.getLogger('server')


# 初始化服务
flask_app = Flask(__name__)
flask_app.add_url_rule(rule='/HeartBeat', view_func=heart_beat, methods=['GET', 'POST'], endpoint='HeartBeat')
flask_app.add_url_rule(rule='/register/cnn_voicemail', view_func=register_cnn_voicemail_view_func,
                       methods=['GET', 'POST'], endpoint='RegisterCnnVoicemail')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        default=settings.port,
        type=int,
    )
    args = parser.parse_args()

    logger.info('model server is already, port: {}'.format(args.port))

    settings.scheduler.init_app(flask_app)
    settings.scheduler.start()

    # settings.scheduler.add_job(
    #     id='scheduler_heartbeat',
    #     func=lambda: logger.debug('scheduler heartbeat'),
    #     trigger='interval', seconds=1
    # )

    # flask_app.run(
    #     host='0.0.0.0',
    #     port=args.port,
    # )

    server = pywsgi.WSGIServer(
        listener=('0.0.0.0', args.port),
        application=flask_app
    )
    server.serve_forever()
