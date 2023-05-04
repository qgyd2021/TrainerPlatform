#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from glob import glob
import logging
import os
import sys

from flask import render_template, request
import jsonschema
import torch
import time

from server.exception import ExpectedError
from server.flask_server.route_wrap.common_route_wrap import common_route_wrap
from server.train_model_server.schema.register import register_cnn_voicemail_schema
from toolbox.logging.misc import json_2_str
from toolbox.os.command import Command
from server.train_model_server import settings
from project_settings import project_path

logger = logging.getLogger('server')


task_cnn_voicemail_to_last_count = defaultdict(int)


def task_cnn_voicemail_func(task_name, language, increase_number, data_dir):
    global task_cnn_voicemail_to_last_count

    last_count = task_cnn_voicemail_to_last_count[task_name]

    filename_pattern = os.path.join(data_dir, 'wav_finished/*/*.wav')
    filename_list = glob(filename_pattern)

    logger.debug('task cnn voicemail, task_name: {}, language: {}, '
                 'increase_number: {}, last_count: {}, this_count: {}, data_dir: {}'.format(
        task_name, language, increase_number, last_count, len(filename_list), data_dir))

    if len(filename_list) > last_count + increase_number:
        task_work_dir = os.path.join(project_path, 'examples/voicemail_classification')

        task_cnn_voicemail_to_last_count[language] = len(filename_list)

        logger.info('run {}'.format(task_name))
        cmd = """cd {task_work_dir} && nohup \
sh run.sh \
--stage -1 --stop_stage 9 \
--system_version {system_version} \
--filename_patterns {filename_pattern1} \
--file_folder_name {file_folder_name} \
--final_model_name {final_model_name} \
&""".format(
            task_work_dir=task_work_dir,
            system_version='centos',
            filename_pattern1=filename_pattern,
            file_folder_name=task_name,
            final_model_name=task_name,
        ).strip()

        logger.info(cmd)
        if sys.platform not in ('win32', ):
            Command.system(cmd)

        return True
    return False


@common_route_wrap
def register_cnn_voicemail_view_func():
    args = request.json
    logger.info('args: {}'.format(json_2_str(args)))

    # 请求体校验
    try:
        jsonschema.validate(args, register_cnn_voicemail_schema)
    except (jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError, ) as e:
        raise ExpectedError(
            status_code=60401,
            message='request body invalid. ',
            detail=str(e)
        )

    language = args['language']
    increase_number = args['increase_number']
    data_dir = args.get('data_dir')
    if data_dir is None:
        data_dir = os.path.join(settings.dataset_dir, language)
    interval = args.get('interval', 24 * 60 * 60)

    task_name = 'task_cnn_voicemail_{}'.format(language)
    settings.scheduler.add_job(
        id=task_name, func=task_cnn_voicemail_func, args=[task_name, language, increase_number, data_dir],
        trigger='interval',
        seconds=interval,
    )

    return task_name


if __name__ == '__main__':
    pass
