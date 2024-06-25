#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import os
import sqlite3
from typing import List

from flask_apscheduler import APScheduler

from project_settings import project_path
from toolbox.os.environment import EnvironmentManager

log_directory = os.path.join(project_path, 'server/train_model_server/logs')
os.makedirs(log_directory, exist_ok=True)

images_directory = os.path.join(project_path, 'server/train_model_server/static/images')
os.makedirs(images_directory, exist_ok=True)

environment = EnvironmentManager(
    path=os.path.join(project_path, 'server/train_model_server/dotenv'),
    env=os.environ.get('environment', 'dev'),
)

port = environment.get(key='port', default=9180, dtype=int)

# task
task_cnn_voicemail_json_settings_file = environment.get(
    key='task_cnn_voicemail_json_settings_file',
    default=os.path.join(project_path, 'server/train_model_server/json_settings/task_cnn_voicemail.json'),
    dtype=str
)
task_cnn_voicemail_common_json_settings_file = environment.get(
    key='task_cnn_voicemail_common_json_settings_file',
    default=os.path.join(project_path, 'server/train_model_server/json_settings/task_cnn_voicemail_common.json'),
    dtype=str
)
task_basic_intent_json_settings_file = environment.get(
    key='task_basic_intent_json_settings_file',
    default=os.path.join(project_path, 'server/train_model_server/json_settings/task_basic_intent.json'),
    dtype=str
)

trained_models_dir = environment.get(
    key='trained_models_dir',
    default=os.path.join(project_path, 'trained_models'),
    dtype=str
)

# plugin
scheduler = APScheduler()


if __name__ == '__main__':
    pass
