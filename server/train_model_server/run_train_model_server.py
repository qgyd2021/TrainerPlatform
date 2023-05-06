#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime, timedelta
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
from server.train_model_server.view_func.cnn_voicemail import cnn_voicemail_view_func, cnn_voicemail_pivot_table_view_func
from server.train_model_server.tasks.task_cnn_voicemail import TaskCnnVoicemailFunc

logger = logging.getLogger('server')


# 初始化服务
flask_app = Flask(__name__)
flask_app.add_url_rule(rule='/HeartBeat', view_func=heart_beat, methods=['GET', 'POST'], endpoint='HeartBeat')
flask_app.add_url_rule(rule='/cnn_voicemail', view_func=cnn_voicemail_view_func, methods=['GET', 'POST'], endpoint='CnnVoicemail')
flask_app.add_url_rule(rule='/cnn_voicemail_pivot_table', view_func=cnn_voicemail_pivot_table_view_func, methods=['GET', 'POST'], endpoint='CnnVoicemailPivotTable')


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

    # TODO:
    # settings.scheduler.add_job(
    #     id='task_cnn_voicemail',
    #     func=TaskCnnVoicemailFunc(),
    #     trigger='interval',
    #     seconds=5,
    #     # seconds=5 * 60 * 60,
    #     next_run_time=datetime.now() + timedelta(seconds=5)
    # )

    # run on 04:00:00 each day.
    settings.scheduler.add_job(
        id='task_cnn_voicemail',
        func=TaskCnnVoicemailFunc(),
        trigger='cron',
        day_of_week='0-6',
        hour=4,
    )

    # flask_app.run(
    #     host='0.0.0.0',
    #     port=args.port,
    # )

    server = pywsgi.WSGIServer(
        listener=('0.0.0.0', args.port),
        application=flask_app
    )
    server.serve_forever()
