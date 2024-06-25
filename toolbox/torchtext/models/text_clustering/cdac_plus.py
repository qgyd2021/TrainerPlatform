#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
from torch.nn import functional

from toolbox.torch.modules.loss import FocalLoss, HingeLoss, HingeLinear
from toolbox.torch.training.metrics.categorical_accuracy import CategoricalAccuracy


class StudentsTDistribution(object):

    @staticmethod
    def demo1():
        samples = np.random.random(size=[2, 5])
        clusters = np.random.random(size=[3, 5])
        samples = torch.tensor(samples)
        clusters = torch.tensor(clusters)

        probs = StudentsTDistribution.q(samples, clusters)
        print(probs)
        return

    @staticmethod
    def q(samples: torch.Tensor,
          clusters: torch.Tensor,
          alpha: float = 1.0
          ):
        """
        :param samples: shape=[n_samples, n_dim]
        :param clusters: shape=[n_cluster, n_dim]
        :param alpha: float
        :return: probs, shape=[n_samples, n_cluster]
        """
        # shape = [n_samples, 1, n_dim]
        samples = torch.unsqueeze(samples, dim=1)
        # shape = [1, n_cluster, n_dim]
        clusters = torch.unsqueeze(clusters, dim=0)

        # shape = [n_samples, n_cluster]
        logits = 1 / (1 + torch.sum(torch.square(samples - clusters), dim=-1) / alpha)
        logits = torch.pow(logits, (alpha + 1.0) / 2.0)
        probs = logits / torch.sum(logits, dim=-1, keepdim=True)
        return probs


class AuxiliaryTargetDistribution(object):

    @staticmethod
    def demo1():
        probs = np.random.random(size=[2, 3])
        probs = torch.tensor(probs)

        target = AuxiliaryTargetDistribution.p(probs)
        print(target)
        return

    @staticmethod
    def p(probs: torch.Tensor):
        """
        :param probs: shape=[n_samples, n_cluster]
        :return:
        """
        weight = torch.square(probs) / torch.sum(probs, dim=0)
        probs = weight / torch.sum(weight, dim=1, keepdim=True)
        return probs


class BertForConstrainClustering(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForConstrainClustering, self).__init__(config)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                ):
        encoded_layer_12, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        # Pooling-mean
        pooled_output = torch.mean(encoded_layer_12, dim=1)
        return pooled_output


class CDACPlus(nn.Module):
    """
    https://arxiv.org/abs/1911.08891


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

    eps = 1e-7

    def __init__(self,
                 backbone,
                 hidden_size: int,
                 dropout: float,
                 n_clusters: int,
                 positive_weight: float = 1.0,
                 negative_weight: float = 1.0
                 ):
        super(CDACPlus, self).__init__()
        self.model = backbone

        self.n_clusters = n_clusters
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

        # train (labeled and unlabeled)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_clusters)

        # finetune (clustering)
        self.alpha = 1.0
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_clusters))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.focal_loss = FocalLoss(
            num_classes=n_clusters,
            reduction='mean',
        )
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                input_ids: torch.LongTensor,
                ):
        pooled_output = self.model.forward(input_ids)
        pooled_output = self.linear(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def supervised_training(self,
                            input_ids: torch.LongTensor,
                            label_ids: torch.LongTensor,
                            upper_threshold: float,
                            lower_threshold: float,
                            ):
        logits = self.forward(input_ids)
        logits_norm = functional.normalize(logits, p=2, dim=1)

        similar_matrix = torch.matmul(logits_norm, logits_norm.transpose(0, -1))

        label_matrix = label_ids.view(-1, 1) - label_ids.view(1, -1)
        label_matrix = torch.where(label_matrix == 0, 1, 0)

        positive_mask = label_matrix > upper_threshold
        negative_mask = label_matrix < lower_threshold

        loss = self.kl_loss(similar_matrix, positive_mask, negative_mask)

        outputs = {
            'loss': loss * 5,
        }
        return outputs

    def semi_supervised_training(self,
                                 input_ids: torch.LongTensor,
                                 label_ids: torch.LongTensor,
                                 upper_threshold: float,
                                 lower_threshold: float,
                                 unknown_label: int,
                                 ):
        logits = self.forward(input_ids)
        logits_norm = functional.normalize(logits, p=2, dim=1)
        similar_matrix = torch.matmul(logits_norm, logits_norm.transpose(0, -1))

        label_matrix = label_ids.view(-1, 1) - label_ids.view(1, -1)
        label_matrix = torch.where(label_matrix == 0, 1, 0)
        label_matrix[label_ids == unknown_label, :] = -1
        label_matrix[:, label_ids == unknown_label] = -1

        positive_mask = similar_matrix > upper_threshold
        negative_mask = similar_matrix < lower_threshold
        positive_mask[label_matrix == 1] = 1
        negative_mask[label_matrix == 0] = 1
        loss = self.kl_loss(similar_matrix, positive_mask, negative_mask)

        outputs = {
            'loss': loss + (upper_threshold - lower_threshold)
        }
        return outputs

    def kl_loss(self,
                similar_matrix: torch.Tensor,
                positive_mask: torch.Tensor,
                negative_mask: torch.Tensor,
                ):
        pos_entropy = -torch.log(torch.clamp(similar_matrix, self.eps, 1.0)) * positive_mask
        neg_entropy = -torch.log(torch.clamp(1 - similar_matrix, self.eps, 1.0)) * negative_mask

        loss = pos_entropy.mean() * self.positive_weight + neg_entropy.mean() * self.negative_weight
        return loss


def demo1():
    StudentsTDistribution.demo1()
    AuxiliaryTargetDistribution.demo1()
    return


if __name__ == '__main__':
    demo1()
