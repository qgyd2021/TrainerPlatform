#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import os
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


class CnnVoicemailService(object):
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

            self.models.set(language, {
                'model': model,
                'vocabulary': vocabulary,
            })

            shutil.rmtree(tgt_path)

        return

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


_cnn_voicemail_service = None


def get_cnn_voicemail_service_instance():
    global _cnn_voicemail_service
    if _cnn_voicemail_service is None:
        _cnn_voicemail_service = CnnVoicemailService(
            trained_models_dir=settings.trained_models_dir
        )
    return _cnn_voicemail_service


if __name__ == '__main__':
    pass
