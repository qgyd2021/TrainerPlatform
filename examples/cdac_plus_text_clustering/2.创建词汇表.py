#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from project_settings import project_path
from toolbox.torch.utils.data.dataset.text_classifier_json_dataset import TextClassifierJsonDataset
from toolbox.torch.utils.data.tokenizers.pretrained_bert_tokenizer import PretrainedBertTokenizer
from toolbox.torch.utils.data.vocabulary import Vocabulary


model_name = r'D:\Users\tianx\PycharmProjects\PyTorch\pretrained\chinese-bert-wwm-ext'

tokenizer = PretrainedBertTokenizer(model_name)

vocabulary = Vocabulary(non_padded_namespaces=['tokens', 'labels'])

with open('train.json', 'r', encoding='utf-8') as f:
    for row in f:
        row = json.loads(row)
        label = row['label']
        vocabulary.add_token_to_namespace(label, namespace='labels')

with open('valid.json', 'r', encoding='utf-8') as f:
    for row in f:
        row = json.loads(row)
        label = row['label']
        vocabulary.add_token_to_namespace(label, namespace='labels')


vocabulary.set_from_file(
    filename=os.path.join(model_name, 'vocab.txt'),
    # filename='/data/tianxing/PycharmProjects/pretrained/chinese-bert-wwm-ext/vocab.txt',
    is_padded=False,
    oov_token='[UNK]',
    namespace='tokens',
)
vocabulary.save_to_files('vocabulary')


if __name__ == '__main__':
    pass
