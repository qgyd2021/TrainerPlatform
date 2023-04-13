#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Block2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1
                 ):
        super(Block2D, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

    def forward(self, inputs: torch.Tensor):
        return self.block(inputs)


class LienarSoftPool(nn.Module):
    """
    Linear softmax, takes logits and returns a probability, near to the actual maximum value.

    A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050
    """
    def __init__(self, dim: int = 1):
        super(LienarSoftPool, self).__init__()
        self.dim = dim

    def forward(self, logits: torch.Tensor, time_decision):
        result = (time_decision**2).sum(self.dim) / time_decision.sum(self.dim)
        return result


class MeanPool(nn.Module):
    def __init__(self, dim: int = 1):
        super(MeanPool, self).__init__()
        self.dim = dim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.dim)


class CRNN(nn.Module):
    """
    https://arxiv.org/abs/2003.12222
    https://github.com/richermans/gpv

    采用 Audioset 数据集
    https://ieeexplore.ieee.org/document/7952261/
    https://research.google.com/audioset/download.html
    https://github.com/audioset/ontology

    https://github.com/jaysimon/audioset_download
    https://www.jianshu.com/p/c4fc899342f7

    模型对片段级别音频 (弱标签 weak label) 做分类.

    """
    def __init__(self, input_dim: int = 64, num_classes: int = 2):
        super(CRNN, self).__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3)
        )
        # 计算 features 的输出维度
        with torch.no_grad():
            rnn_input_dim = self.features(
                torch.randn(1, 1, 500, input_dim)
            ).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        # 将其改为单向传播,可实现流式预测.
        self.gru = nn.GRU(
            rnn_input_dim,
            128,
            bidirectional=True,
            batch_first=True,
        )
        self.pooling = LienarSoftPool(dim=1)

        self.outputlayer = nn.Linear(256, num_classes)

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: shape = [batch_size, seq_length, spec_dim]
        :return:
        """
        batch_size, seq_length, spec_dim = inputs.shape
        x = inputs.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)

        x, _ = self.gru(x)

        # sigmoid 音频帧可包含多种声音,使模型学习到各种声音.
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)

        # 插值,cnn 池化后序列长度减小,此处插值到原来的长度.
        decision_time = torch.nn.functional.interpolate(
            decision_time.transpose(1, 2),
            seq_length,
            mode='linear',
            align_corners=False,
        ).transpose(1, 2)

        # 将序列池化为单个向量,用做对整段音频的分类.
        decision = self.pooling(x, decision_time).clamp(1e-7, 1.).squeeze(1)
        return decision, decision_time


if __name__ == '__main__':
    pass
