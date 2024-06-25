#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, List
import zipfile

import numpy as np
import pandas as pd
import torch

from server.exception import ExpectedError
from server.train_model_server import settings
from toolbox.torch.utils.data.vocabulary import Vocabulary
from toolbox.torch.utils.data.tokenizers.pretrained_bert_tokenizer import PretrainedBertTokenizer

logger = logging.getLogger('server')


class BasicIntentByLanguageService(object):
    def __init__(self,
                 trained_models_dir: str
                 ):
        self.trained_models_dir = Path(trained_models_dir)
        self.models = dict()

    @staticmethod
    def get_final_model_name(language: str):
        """basic_intent_en"""
        language_map = {
            'chinese': 'cn',
            'english': 'en',
            'japanese': 'jp',
            'vietnamese': 'vi'
        }
        # language = language.split('-')[0].lower()
        language = language_map[language]
        result = 'basic_intent_{}'.format(language)
        return result

    def load_model(self, language: str):
        zip_file = self.trained_models_dir / '{}.zip'.format(self.get_final_model_name(language))
        if not os.path.exists(zip_file):
            logger.info('no basic intent model for language: {}'.format(language))
            raise ExpectedError(
                status_code=60401,
                message='invalid language: {}'.format(language),
            )
        logger.info('loading basic intent model of language: {}'.format(language))
        with zipfile.ZipFile(zip_file, 'r') as f_zip:
            out_root = Path(tempfile.gettempdir()) / 'basic_intent'
            out_root.mkdir(parents=True, exist_ok=True)
            tgt_path = out_root / self.get_final_model_name(language)
            f_zip.extractall(path=out_root)
            model = torch.jit.load((tgt_path / 'final.zip').as_posix())
            vocabulary = Vocabulary.from_files((tgt_path / 'vocabulary').as_posix())
            tokenizer = PretrainedBertTokenizer(tgt_path.as_posix())

            evaluation = pd.read_excel((tgt_path / 'test_output.xlsx').as_posix())

            # pivot_table
            evaluation = evaluation[evaluation['selected'] == 1]
            pivot_table_ = pd.pivot_table(evaluation, index=['label1'], values=['correct'], aggfunc=['count', 'mean'])
            pivot_table_ = pivot_table_.to_dict()

            pivot_table = dict()
            for row in pivot_table_[('count', 'correct')].items():
                pivot_table[row[0]] = [
                    row[0],
                    round(row[1], 4)
                ]
            for row in pivot_table_[('mean', 'correct')].items():
                pivot_table[row[0]].append(round(row[1], 4))

            pivot_table = list(sorted(pivot_table.values(), key=lambda x: x[1], reverse=True))

            self.models[language] = {
                'model': model,
                'vocabulary': vocabulary,
                'tokenizer': tokenizer,
                'pivot_table': pivot_table,
            }

            shutil.rmtree(tgt_path)

    def get_pivot_table(self, language: str):
        m = self.models.get(language)
        if m is None:
            self.load_model(language)
            m = self.models.get(language)

        pivot_table = m['pivot_table']
        return pivot_table

    def forward(self, text: str, language: str) -> str:
        m = self.models.get(language)
        if m is None:
            self.load_model(language)
            m = self.models.get(language)

        model = m['model']
        vocabulary = m['vocabulary']
        tokenizer = m['tokenizer']

        tokens: List[str] = tokenizer.tokenize(text)
        tokens: List[int] = [vocabulary.get_token_index(token, namespace='tokens') for token in tokens]
        tokens = vocabulary.pad_or_truncate_ids_by_max_length(tokens, max_length=5)
        batch_tokens = [tokens]
        batch_tokens = torch.from_numpy(np.array(batch_tokens))

        outputs = model.forward(batch_tokens)

        probs = outputs['probs']
        argmax = torch.argmax(probs, dim=-1)
        probs = probs.tolist()[0]
        argmax = argmax.tolist()[0]

        # probs = outputs['probs']
        # label_idx = torch.argmax(probs, dim=-1)
        # label_idx = label_idx.numpy()
        # prob = probs.tolist[0][label_idx]

        label_str = vocabulary.get_token_from_index(argmax, namespace='labels')
        prob = probs[argmax]

        result = {
            'label': label_str,
            'prob': round(prob, 4)
        }
        return result

    def release_cache(self):
        self.models = dict()


_basic_intent_by_language_service = None


def get_basic_intent_by_language_service_instance():
    global _basic_intent_by_language_service
    if _basic_intent_by_language_service is None:
        _basic_intent_by_language_service = BasicIntentByLanguageService(
            trained_models_dir=settings.trained_models_dir
        )
    return _basic_intent_by_language_service


if __name__ == '__main__':
    pass
