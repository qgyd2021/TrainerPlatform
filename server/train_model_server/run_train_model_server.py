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
from server.train_model_server.view_func.basic_intent import basic_intent_by_language_view_func, \
    basic_intent_by_language_pivot_table_view_func
from server.train_model_server.view_func.cnn_voicemail import cnn_voicemail_by_language_view_func, \
    cnn_voicemail_by_language_pivot_table_view_func, \
    cnn_voicemail_common_view_func, cnn_voicemail_view_func, cnn_voicemail_correction_view_func
from server.train_model_server.tasks.task_basic_intent import TaskBasicIntentFunc
from server.train_model_server.tasks.task_cnn_voicemail import TaskCnnVoicemailFunc, TaskCnnVoicemailCommonFunc
from server.train_model_server.service.basic_intent import get_basic_intent_by_language_service_instance
from server.train_model_server.service.cnn_voicemail import get_cnn_voicemail_common_service_instance, \
    get_cnn_voicemail_by_language_service_instance

logger = logging.getLogger('server')


# 初始化服务
flask_app = Flask(__name__)
flask_app.add_url_rule(rule='/HeartBeat', view_func=heart_beat, methods=['GET', 'POST'], endpoint='HeartBeat')
flask_app.add_url_rule(rule='/cnn_voicemail_by_language', view_func=cnn_voicemail_by_language_view_func, methods=['GET', 'POST'], endpoint='CnnVoicemailByLanguage')
flask_app.add_url_rule(rule='/cnn_voicemail_by_language_pivot_table', view_func=cnn_voicemail_by_language_pivot_table_view_func, methods=['GET', 'POST'], endpoint='CnnVoicemailByLanguagePivotTable')
flask_app.add_url_rule(rule='/cnn_voicemail_common', view_func=cnn_voicemail_common_view_func, methods=['GET', 'POST'], endpoint='CnnVoicemailCommon')
flask_app.add_url_rule(rule='/cnn_voicemail', view_func=cnn_voicemail_view_func, methods=['GET', 'POST'], endpoint='CnnVoicemail')
flask_app.add_url_rule(rule='/cnn_voicemail_correction', view_func=cnn_voicemail_correction_view_func, methods=['GET', 'POST'], endpoint='CnnVoicemailCorrection')

flask_app.add_url_rule(rule='/basic_intent_by_language', view_func=basic_intent_by_language_view_func, methods=['GET', 'POST'], endpoint='BasicIntentByLanguage')
flask_app.add_url_rule(rule='/basic_intent_by_language_pivot_table', view_func=basic_intent_by_language_pivot_table_view_func, methods=['GET', 'POST'], endpoint='BasicIntentByLanguagePivotTable')


settings.scheduler.init_app(flask_app)
settings.scheduler.start()


def release_cache():
    get_basic_intent_by_language_service_instance().release_cache()
    get_cnn_voicemail_common_service_instance().release_cache()
    get_cnn_voicemail_by_language_service_instance().release_cache()


settings.scheduler.add_job(
    id='task_release_cache',
    func=release_cache,
    trigger='interval',
    seconds=1 * 60 * 60,
    # next_run_time=datetime.now() + timedelta(seconds=5)
)
# run on 02:30:00 each day.
settings.scheduler.add_job(
    id='task_basic_intent',
    func=TaskBasicIntentFunc(),
    trigger='cron',
    day_of_week='0-6',
    hour=2,
    minute=30,
    # next_run_time=datetime.now() + timedelta(seconds=5)
)
# run on 03:00:00 each day.
settings.scheduler.add_job(
    id='task_cnn_voicemail',
    func=TaskCnnVoicemailFunc(),
    trigger='cron',
    day_of_week='0-6',
    hour=3,
    next_run_time=datetime.now() + timedelta(seconds=5)
)
# run on 04:00:00 each day.
settings.scheduler.add_job(
    id='task_cnn_voicemail_common',
    func=TaskCnnVoicemailCommonFunc(),
    trigger='cron',
    day_of_week='0-6',
    hour=3,
    minute=30,
    # next_run_time=datetime.now() + timedelta(seconds=5)
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        default=settings.port,
        type=int,
    )
    args = parser.parse_args()

    logger.info('model server is already, port: {}'.format(args.port))

    # flask_app.run(
    #     host='0.0.0.0',
    #     port=args.port,
    # )

    server = pywsgi.WSGIServer(
        listener=('0.0.0.0', args.port),
        application=flask_app
    )
    server.serve_forever()
