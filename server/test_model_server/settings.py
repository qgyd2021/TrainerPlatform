#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import os
import sqlite3
from typing import List

from project_settings import project_path
from toolbox.os.environment import EnvironmentManager

log_directory = os.path.join(project_path, 'server/test_model_server/logs')
os.makedirs(log_directory, exist_ok=True)

images_directory = os.path.join(project_path, 'server/test_model_server/static/images')
os.makedirs(images_directory, exist_ok=True)

environment = EnvironmentManager(
    path=os.path.join(project_path, 'server/test_model_server/dotenv'),
    env=os.environ.get('environment', 'dev'),
)

port = environment.get(key='port', default=9180, dtype=int)

models_dir = os.path.join(project_path, 'models')
