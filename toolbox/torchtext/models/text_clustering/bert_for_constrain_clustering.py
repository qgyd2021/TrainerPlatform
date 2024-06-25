#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn


class StudentsTDistribution(object):

    @staticmethod
    def demo1():
        q = StudentsTDistribution()

        samples = np.random.random(size=[2, 5])
        clusters = np.random.random(size=[3, 5])

        probs = q.q(samples, clusters)
        print(probs)
        return

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def q(self,
          samples: np.ndarray,
          clusters: np.ndarray,
          ):
        """

        :param samples: shape=[n_samples, n_dim]
        :param clusters: shape=[n_cluster, n_dim]
        :return: probs, shape=[n_samples, n_cluster]
        """
        # shape = [n_samples, 1, n_dim]
        samples = np.expand_dims(samples, axis=1)
        # shape = [1, n_cluster, n_dim]
        clusters = np.expand_dims(clusters, axis=0)

        # shape = [n_samples, n_cluster]
        logits = 1 / (1 + np.sum(np.square(samples - clusters), axis=-1) / self.alpha)
        logits = np.power(logits, (self.alpha + 1.0) / 2.0)
        probs = logits / np.sum(logits, axis=-1, keepdims=True)
        return probs


class AuxiliaryTargetDistribution(object):
    @staticmethod
    def demo1():
        probs = np.random.random(size=[2, 3])
        target = AuxiliaryTargetDistribution.p(probs)
        print(target)
        return

    @staticmethod
    def p(probs: np.ndarray):
        """
        :param probs: shape=[n_samples, n_cluster]
        :return:
        """
        weight = np.square(probs) / np.sum(probs, axis=0)
        probs = weight / np.sum(weight, axis=1, keepdims=True)
        return probs


class BertForConstrainClustering(BertPreTrainedModel):
    """
    (1)预训练
    使用所有的已标注和未标注数据.
    BertModel 只开放最后一层, 的参数优化.

    已标注数据训练
    mini-batch 内,
    相同标注的数据为相似的句子.
    不同的都认为是不相似的句子.

    全部数据训练
    mini-batch 内,
    计算句子 logits 余弦相似度.
    相同标注的数据为相似的句子.
    不同的都认为是不相似的句子.
    未标注数据中, 余弦相似度, 大于 upper 阈值的认为相似, 小于 lower 阈值的认为不相似.

    已标注数据, 全部数据, 交替训练,



    (2)聚类微调
    开放 BertModel 全部的参数优化.

    模型生成句向量, 作 KMeans 聚类.
    将聚类中心向量, 写入 self.cluster_layer.

    模型训练, KL散度损失.


    """
    def __init__(self, config, num_labels):
        super(BertForConstrainClustering, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)

        # train (labeled and unlabeled)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # Pooling-mean
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

        # finetune (clustering)
        self.alpha = 1.0
        self.cluster_layer = nn.Parameter(torch.Tensor(num_labels, num_labels))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                u_threshold=None,
                l_threshold=None,
                mode=None,
                labels=None,
                semi=False
                ):
        eps = 1e-10
        encoded_layer_12, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # Pooling-mean
        pooled_output = self.dense(encoded_layer_12.mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def demo1():
    StudentsTDistribution.demo1()
    AuxiliaryTargetDistribution.demo1()
    return


if __name__ == '__main__':
    demo1()
