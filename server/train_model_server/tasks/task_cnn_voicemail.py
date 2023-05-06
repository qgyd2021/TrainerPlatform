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

from toolbox.os.command import Command
from project_settings import project_path
from server.train_model_server import settings

logger = logging.getLogger('server')


class TaskCnnVoicemailFunc(object):
    def __init__(self):
        self.task_cnn_voicemail_to_last_count = defaultdict(int)

        self.dataset_dir: Path = None
        self.languages: List[str] = list()

    @staticmethod
    def get_file_folder_name(language: str):
        language = language.replace('-', '_').lower()
        result = 'file_dir_{}'.format(language)
        return result

    @staticmethod
    def get_final_model_name(language: str):
        language = language.replace('-', '_').lower()
        result = 'cnn_voicemail_{}'.format(language)
        return result

    @staticmethod
    def get_nohup_name(language: str):
        language = language.replace('-', '_').lower()
        result = 'nohup_{}.out'.format(language)
        return result

    def read_cnn_voicemail_settings(self, settings_file: str):
        with open(settings_file, 'rb') as f:
            cnn_voicemail_settings = json.load(f)

        self.dataset_dir = Path(cnn_voicemail_settings['dataset_dir'])
        tasks = cnn_voicemail_settings['tasks']
        for task in tasks:
            language = task['language']
            start_count = task['start_count']
            self.task_cnn_voicemail_to_last_count[language] = start_count
            self.languages.append(language)

    def __call__(self):
        self.read_cnn_voicemail_settings(settings_file=settings.task_cnn_voicemail_json_settings_file)

        for language in self.languages:
            filename_pattern = self.dataset_dir / language / 'wav_finished/*/*.wav'
            filename_pattern = str(filename_pattern)
            filename_list = glob(filename_pattern)

            last_count = self.task_cnn_voicemail_to_last_count[language]
            this_count = len(filename_list)

            logger.debug(
                'task cnn voicemail, language: {}, '
                'last_count: {}, this_count: {}'.format(
                    language, last_count, len(filename_list)))

            if this_count - last_count > 5000:
                task_work_dir = os.path.join(project_path, 'examples/voicemail_classification')

                self.task_cnn_voicemail_to_last_count[language] = len(filename_list)

                cmd = """nohup \
                    sh run.sh \
                    --stage -1 --stop_stage 9 \
                    --system_version {system_version} \
                    --filename_patterns {filename_pattern1} \
                    --file_folder_name {file_folder_name} \
                    --final_model_name {final_model_name} \
                    > {nohup_name} &""".format(
                    system_version='centos',
                    filename_pattern1=filename_pattern.replace(r'*', r'\*'),
                    language=language.replace('-', '_').lower(),
                    file_folder_name=self.get_file_folder_name(language),
                    final_model_name=self.get_final_model_name(language),
                    nohup_name=self.get_nohup_name(language)
                ).strip()
                cmd = re.sub(r'[\u0020]{4,}', ' ', cmd)

                logger.info(cmd)
                if sys.platform not in ('win32', ):
                    Command.cd(task_work_dir)
                    Command.system(cmd)


if __name__ == '__main__':
    pass
