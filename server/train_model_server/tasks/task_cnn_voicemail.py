#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from glob import glob
import json
import logging
import os
from pathlib import Path
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
        self.languages: List[str] = None

    def read_cnn_voicemail_settings(self, settings_file: str):
        with open(settings_file, 'rb') as f:
            cnn_voicemail_settings = json.load(f)

        self.dataset_dir = Path(cnn_voicemail_settings['dataset_dir'])
        self.languages = cnn_voicemail_settings['languages']

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
                    --file_folder_name {language} \
                    --final_model_name {language} \
                    > nohup_{language}.out &""".format(
                    system_version='centos',
                    filename_pattern1=filename_pattern.replace(r'*', r'\*'),
                    language=language.replace('-', '_').lower()
                ).strip()

                logger.info(cmd)
                if sys.platform not in ('win32', ):
                    Command.cd(task_work_dir)
                    Command.system(cmd)


if __name__ == '__main__':
    pass
