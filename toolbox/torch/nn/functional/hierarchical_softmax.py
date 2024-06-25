#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import OrderedDict
import pickle
from typing import Dict, Optional

from overrides import overrides
import torch
import torch.nn as nn


class HierarchicalSoftMax(nn.Module):
    """多层 softmax 实现多极文本分类

    由于初始化时, 各层 softmax 的概率趋于平衡.

    因此在第一层时 `领域无关` 就分到了 50% 的概率.

    `领域相关` 中的各类别去分剩下的 50% 的概率.
    这会导致模型一开始时输出的类别全是 `领域无关`, 这导致模型无法优化.

    解决方案:
    1. 从数据集中去除 `领域无关` 数据. 并训练模型.
    2. 等模型收敛之后, 再使用包含 `领域无关` 的数据集, 让模型加载之前的权重, 并重新开始训练模型.

    """

    @staticmethod
    def demo1():
        # hierarchical_labels = OrderedDict({
        #     '领域相关': OrderedDict({
        #         '肯定答复': [
        #             '肯定(好的)', '肯定(可以)', '肯定(正确)'
        #         ],
        #         '否定答复': [
        #             '否定(不可以)', '否定(不知道)', '否定(错误)'
        #         ],
        #         '用户正忙': [
        #             '用户正忙'
        #         ]
        #     }),
        #     '领域无关': OrderedDict({
        #         '领域无关': [
        #             '领域无关'
        #         ]
        #     })
        # })

        hierarchical_labels = OrderedDict({
            '领域相关': ['肯定答复', '否定答复', '用户正忙', '查联系方式'],
            '领域无关': ['领域无关'],
        })

        softmax_layer = HierarchicalSoftMax(
            classifier_input_dim=3,
            hierarchical_labels=hierarchical_labels,
            activation='softmax',
            # activation='sigmoid',

        )

        for k, v in softmax_layer.__dict__['_modules'].items():
            print(k)
            print(v)

        inputs = torch.ones(size=(2, 3), dtype=torch.float32)

        probs = softmax_layer.forward(inputs)
        print(probs)
        print(torch.sum(probs, dim=-1))
        return

    def __init__(self, classifier_input_dim: int, hierarchical_labels: OrderedDict, activation: str = 'softmax'):
        super(HierarchicalSoftMax, self).__init__()
        self.classifier_input_dim = classifier_input_dim
        self.hierarchical_labels = hierarchical_labels
        self.activation: str = activation

        self._init_hierarchical_classification_layer(hierarchical_labels)

    def _init_hierarchical_classification_layer(self,
                                                hierarchical_labels: OrderedDict,
                                                key: str = 'classification_layer',
                                                child_class: str = None):
        num_labels = len(hierarchical_labels)

        classification_layer = torch.nn.Linear(self.classifier_input_dim, num_labels)
        if child_class is not None:
            key = '{header}_{child_class}'.format(header=key, child_class=child_class)
        setattr(
            self,
            key,
            classification_layer
        )

        branch = 0
        for k, v in hierarchical_labels.items():
            if isinstance(v, OrderedDict):
                self._init_hierarchical_classification_layer(
                    v,
                    key=key,
                    child_class=branch,
                )
            elif isinstance(v, list):
                num_labels = len(v)
                classification_layer = torch.nn.Linear(self.classifier_input_dim, num_labels)
                setattr(
                    self,
                    '{key}_{child_class}'.format(key=key, child_class=branch),
                    classification_layer,
                )
            else:
                raise NotImplementedError
            branch += 1
        return

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        key = 'classification_layer'
        classification_layer = getattr(self, key)
        logits = classification_layer.forward(inputs)
        probs = torch.softmax(logits, dim=-1)

        probs = self._layer_probs(
            inputs=inputs,
            probs=probs,
            key=key,
        )

        return probs

    def _layer_probs(self,
                     inputs: torch.Tensor,
                     probs: torch.Tensor,
                     key: str,
                     ):

        result = list()
        for child_class in range(probs.shape[1]):
            parent_probs = torch.unsqueeze(probs[:, child_class], dim=-1)

            child_key = '{key}_{child_class}'.format(key=key, child_class=child_class)
            classification_layer = getattr(self, child_key)
            logits = classification_layer.forward(inputs)

            child_child_key = '{key}_{child_class}'.format(key=child_key, child_class=0)
            if hasattr(self, child_child_key):
                child_probs = torch.softmax(logits, dim=-1)
                child_probs = child_probs * parent_probs

                child_probs = self._layer_probs(
                    inputs=inputs,
                    probs=child_probs,
                    key=child_key,
                )
            else:
                if self.activation == 'softmax':
                    child_probs = torch.softmax(logits, dim=-1)
                else:
                    child_probs = torch.sigmoid(logits)
                child_probs = child_probs * parent_probs

            result.append(child_probs)

        result = torch.concat(result, dim=-1)
        return result


def demo1():
    HierarchicalSoftMax.demo1()
    return


if __name__ == '__main__':
    demo1()
