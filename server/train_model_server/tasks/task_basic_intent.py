#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from glob import glob
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import List

import pandas as pd

from toolbox.os.command import Command
from project_settings import project_path
from server.train_model_server import settings

logger = logging.getLogger('server')


class TaskBasicIntentFunc(object):
    def __init__(self):
        self.task_basic_intent_to_last_count = defaultdict(int)

        self.dataset_dir: Path = None
        self.languages: List[str] = list()

    @staticmethod
    def get_file_folder_name(language: str):
        language = language.lower()
        result = 'file_dir_{}'.format(language)
        return result

    @staticmethod
    def get_final_model_name(language: str):
        language_map = {
            'chinese': 'cn',
            'english': 'en',
        }
        language = language.lower()

        language = language_map[language]

        result = 'basic_intent_{}'.format(language)
        return result

    @staticmethod
    def get_nohup_name(language: str):
        language = language.lower()
        result = 'nohup_{}.out'.format(language)
        return result

    @staticmethod
    def get_pretrained_bert_model_name(language: str):
        model_map = {
            'chinese': 'chinese-bert-wwm-ext',
            'english': 'bert-base-uncased',
            'japanese': 'bert-base-japanese',
            'vietnamese': 'bert-base-vietnamese-uncased'
        }
        language = language.lower()
        result = model_map[language]
        return result

    def read_basic_intent_settings(self, settings_file: str):
        with open(settings_file, 'rb') as f:
            basic_intent_settings = json.load(f)

        self.dataset_dir = Path(basic_intent_settings['dataset_dir'])
        tasks = basic_intent_settings['tasks']

        languages = list()
        for task in tasks:
            language = task['language']
            start_count = task['start_count']
            if self.task_basic_intent_to_last_count[language] < start_count:
                self.task_basic_intent_to_last_count[language] = start_count
            languages.append(language)
        self.languages = languages

    def __call__(self):
        self.read_basic_intent_settings(settings_file=settings.task_basic_intent_json_settings_file)

        for language in self.languages:
            filename = self.dataset_dir / language / 'dataset.xlsx'

            df = pd.read_excel(filename)
            df = df[df['selected'] == 1]

            last_count = self.task_basic_intent_to_last_count[language]
            this_count = len(df)

            logger.debug(
                'task basic intent, language: {}, '
                'last_count: {}, this_count: {}'.format(
                    language, last_count, this_count))

            if this_count - last_count > 5000:
                task_work_dir = os.path.join(project_path, 'examples/basic_intent_classification')

                self.task_basic_intent_to_last_count[language] = this_count

                cmd = """nohup \
                    sh run.sh \
                    --stage -1 --stop_stage 9 \
                    --system_version {system_version} \
                    --dataset_filename {dataset_filename} \
                    --pretrained_bert_model_name {pretrained_bert_model_name} \
                    --file_folder_name {file_folder_name} \
                    --final_model_name {final_model_name} \
                    > {nohup_name} &""".format(
                    system_version='centos',
                    dataset_filename=filename,
                    pretrained_bert_model_name=self.get_pretrained_bert_model_name(language),
                    file_folder_name=self.get_file_folder_name(language),
                    final_model_name=self.get_final_model_name(language),
                    nohup_name=self.get_nohup_name(language)
                ).strip()
                cmd = re.sub(r'[\u0020]{4,}', ' ', cmd)

                logger.info('cmd: {}'.format(cmd))
                if sys.platform not in ('win32', ):
                    Command.cd(task_work_dir)
                    Command.system(cmd)


if __name__ == '__main__':
    pass
