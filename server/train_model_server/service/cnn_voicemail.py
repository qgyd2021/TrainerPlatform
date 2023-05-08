#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict
import zipfile

from cacheout import Cache
import numpy as np
import torch

from server.train_model_server.tasks.task_cnn_voicemail import TaskCnnVoicemailFunc
from toolbox.torch.utils.data.vocabulary import Vocabulary
from server.train_model_server import settings
from server.exception import ExpectedError

logger = logging.getLogger('server')


class CnnVoicemailByLanguageService(object):
    def __init__(self, trained_models_dir: str):
        self.trained_models_dir = Path(trained_models_dir)
        self.models = Cache(maxsize=256, ttl=1 * 60 * 60, timer=time.time)

    def load_model(self, language: str):
        logger.info('load model: {}'.format(language))
        zip_file = self.trained_models_dir / '{}.zip'.format(TaskCnnVoicemailFunc.get_final_model_name(language))
        if not os.path.exists(zip_file):
            raise ExpectedError(
                status_code=60401,
                message='invalid language: {}'.format(language),
            )
        with zipfile.ZipFile(zip_file, 'r') as f_zip:
            out_root = Path(tempfile.gettempdir()) / 'cnn_voicemail'
            out_root.mkdir(parents=True, exist_ok=True)
            tgt_path = out_root / TaskCnnVoicemailFunc.get_final_model_name(language)
            f_zip.extractall(path=out_root)
            model = torch.jit.load((tgt_path / 'cnn_voicemail.pth').as_posix())
            vocabulary = Vocabulary.from_files((tgt_path / 'vocabulary').as_posix())
            evaluation = pd.read_excel((tgt_path / 'evaluation.xlsx').as_posix())

            pivot_table = pd.pivot_table(evaluation, index=['flag', 'label'], values=['correct'], aggfunc=['mean', 'count'])
            pivot_table = pivot_table.to_dict()

            result = list()
            for row in pivot_table[('mean', 'correct')].items():
                result.append(('_'.join(row[0]), round(row[1], 4)))
            for row in pivot_table[('count', 'correct')].items():
                result.append(('_'.join(row[0]), int(row[1])))

            self.models.set(language, {
                'model': model,
                'vocabulary': vocabulary,
                'pivot_table': result,
            })

            shutil.rmtree(tgt_path)

        return

    def get_pivot_table(self, language: str):
        m = self.models.get(language)
        if m is None:
            self.load_model(language)
            m = self.models.get(language)

        pivot_table = m['pivot_table']
        return pivot_table

    def forward(self, signal: np.ndarray, language: str) -> str:
        m = self.models.get(language)
        if m is None:
            self.load_model(language)
            m = self.models.get(language)

        model = m['model']
        vocabulary = m['vocabulary']

        inputs = torch.tensor(signal, dtype=torch.float32)
        inputs = torch.unsqueeze(inputs, dim=0)

        outputs = model(inputs)

        probs = outputs['probs']
        argmax = torch.argmax(probs, dim=-1)
        probs = probs.tolist()[0]
        argmax = argmax.tolist()[0]

        label = vocabulary.get_token_from_index(argmax, namespace='labels')
        prob = probs[argmax]

        result = {
            'label': label,
            'prob': round(prob, 4)
        }
        return result


_cnn_voicemail_by_language_service = None


def get_cnn_voicemail_by_language_service_instance():
    global _cnn_voicemail_by_language_service
    if _cnn_voicemail_by_language_service is None:
        _cnn_voicemail_by_language_service = CnnVoicemailByLanguageService(
            trained_models_dir=settings.trained_models_dir
        )
    return _cnn_voicemail_by_language_service


class CnnVoicemailCommonService(object):
    def __init__(self, trained_models_dir: str):
        self.trained_models_dir = Path(trained_models_dir)
        self.models = Cache(maxsize=256, ttl=1 * 60 * 60, timer=time.time)

    def load_model(self, key: str = 'default'):
        logger.info('load model')
        zip_file = self.trained_models_dir / 'cnn_voicemail_common.zip'
        if not os.path.exists(zip_file):
            raise ExpectedError(
                status_code=60401,
                message='file not exist: {}'.format(zip_file.as_posix()),
            )
        with zipfile.ZipFile(zip_file, 'r') as f_zip:
            out_root = Path(tempfile.gettempdir()) / 'cnn_voicemail'
            out_root.mkdir(parents=True, exist_ok=True)
            tgt_path = out_root / 'cnn_voicemail_common'
            f_zip.extractall(path=out_root)
            model = torch.jit.load((tgt_path / 'cnn_voicemail.pth').as_posix())
            vocabulary = Vocabulary.from_files((tgt_path / 'vocabulary').as_posix())
            evaluation = pd.read_excel((tgt_path / 'evaluation.xlsx').as_posix())

            pivot_table = pd.pivot_table(evaluation, index=['flag', 'label'], values=['correct'], aggfunc=['mean', 'count'])
            pivot_table = pivot_table.to_dict()

            result = list()
            for row in pivot_table[('mean', 'correct')].items():
                result.append(('_'.join(row[0]), round(row[1], 4)))
            for row in pivot_table[('count', 'correct')].items():
                result.append(('_'.join(row[0]), int(row[1])))

            self.models.set(key, {
                'model': model,
                'vocabulary': vocabulary,
                'pivot_table': result,
            })

            shutil.rmtree(tgt_path)

        return

    def get_pivot_table(self, key: str = 'default'):
        m = self.models.get(key)
        if m is None:
            self.load_model(key)
            m = self.models.get(key)

        pivot_table = m['pivot_table']
        return pivot_table

    def forward(self, signal: np.ndarray, key: str = 'default') -> str:
        m = self.models.get(key)
        if m is None:
            self.load_model(key)
            m = self.models.get(key)

        model = m['model']
        vocabulary = m['vocabulary']

        inputs = torch.tensor(signal, dtype=torch.float32)
        inputs = torch.unsqueeze(inputs, dim=0)

        outputs = model(inputs)

        probs = outputs['probs']
        argmax = torch.argmax(probs, dim=-1)
        probs = probs.tolist()[0]
        argmax = argmax.tolist()[0]

        label = vocabulary.get_token_from_index(argmax, namespace='labels')
        prob = probs[argmax]

        result = {
            'label': label,
            'prob': round(prob, 4)
        }
        return result


_cnn_voicemail_common_service = None


def get_cnn_voicemail_common_service_instance():
    global _cnn_voicemail_common_service
    if _cnn_voicemail_common_service is None:
        _cnn_voicemail_common_service = CnnVoicemailCommonService(
            trained_models_dir=settings.trained_models_dir
        )
    return _cnn_voicemail_common_service


if __name__ == '__main__':
    pass
