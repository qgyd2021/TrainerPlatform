#!/usr/bin/python3
# -*- coding: utf-8 -*-
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert_japanese.tokenization_bert_japanese import BertJapaneseTokenizer
from toolbox.torch.utils.data.tokenizers.tokenizer import Tokenizer


class PretrainedBertTokenizer(Tokenizer):
    def __init__(self, model_name: str, do_lowercase: bool = True):
        if model_name.__contains__('japanese'):
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        self.do_lowercase = do_lowercase

    def tokenize(self, text: str):
        if self.do_lowercase:
            text = str(text).lower()
        tokens = self.tokenizer.tokenize(text)

        tokens = ['[CLS]'] + tokens + ['[SEP]']

        return tokens


if __name__ == '__main__':
    pass
