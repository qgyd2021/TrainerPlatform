#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

from project_settings import project_path
from toolbox.torch.utils.data.dataset.text_classifier_json_dataset import TextClassifierJsonDataset
from toolbox.torch.utils.data.tokenizers.pretrained_bert_tokenizer import PretrainedBertTokenizer
from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--intent_classification_xlsx',
        default='datasets/basic_intent_classification/intent_classification_cn.xlsx',
        type=str
    )
    parser.add_argument('--pretrained_model_dir', default='chinese-bert-wwm-ext', type=str)
    parser.add_argument('--train_all', default='train_all.json', type=str)
    parser.add_argument('--vocabulary', default='vocabulary', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model_name = args.pretrained_model_dir

    vocabulary = Vocabulary(non_padded_namespaces=['tokens', 'labels'])

    with open(args.train_all, 'r', encoding='utf-8') as f:
        for row in f:
            row = json.loads(row)
            label = row['label']
            vocabulary.add_token_to_namespace(label, namespace='labels')

    vocabulary.set_from_file(
        filename=os.path.join(model_name, 'vocab.txt'),
        is_padded=False,
        oov_token='[UNK]',
        namespace='tokens',
    )
    vocabulary.save_to_files(args.vocabulary)

    return


if __name__ == '__main__':
    main()
