#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List, Union, Tuple

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


class Conv1dReluBatchNorm(nn.Module):
    """
    Conv1d + ReLU + BatchNorm1d
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False
                 ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        :param x: shape = [batch_size, channels, seq_length]
        :return:
        """
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class Res2DilatedConv1dReluBatchNorm(nn.Module):
    """
    Res2 Dilated Conv1d + BatchNorm1d + ReLU

    [16] S. Gao, M.-M. Cheng, K. Zhao, X. Zhang, M.-H. Yang, and
    P. H. S. Torr, “Res2Net: A new multi-scale backbone architecture,” IEEE TPAMI, 2019.

    """
    def __init__(self,
                 channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 scale: int = 4
                 ):
        super().__init__()
        if channels % scale != 0:
            raise AssertionError('{} % {} != 0'.format(channels, scale))

        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.conv1d_relu_batch_norm_list = nn.ModuleList(modules=[
            Conv1dReluBatchNorm(
                in_channels=self.width,
                out_channels=self.width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
            for _ in range(self.nums)
        ])

    def forward(self, x):
        """
        :param x: shape = [batch_size, channels, seq_length]
        :return:
        """
        out = []
        spx = torch.split(tensor=x, split_size_or_sections=self.width, dim=1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.conv1d_relu_batch_norm_list[i](sp)

            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])

        out = torch.cat(out, dim=1)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        if channels % s != 0:
            raise AssertionError('{} % {} != 0'.format(channels, s))

        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, inputs):
        """
        :param inputs: shape = [batch_size, spec_dim, seq_length]
        :return:
        """
        x = torch.mean(inputs, dim=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)

        outputs = inputs * torch.unsqueeze(x, dim=2)
        return outputs


class SqueezeExcitationRes2Block(nn.Module):
    """
    [14] J. Hu, L. Shen, and G. Sun, “Squeeze-and-Excitation networks,”
    in Proc. IEEE/CVF CVPR, 2018, pp. 7132–7141.

    [15] J. Zhou, T. Jiang, Z. Li, L. Li, and Q. Hong,
    “Deep speaker embedding extraction with channel-wise feature
    responses and additive supervision softmax loss function,”
    in Proc. Interspeech, 2019, pp. 2883–2887.
    """
    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 dilation: int,
                 scale: int,
                 ):
        super(SqueezeExcitationRes2Block, self).__init__()

        self.layers = nn.Sequential(
            Conv1dReluBatchNorm(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Res2DilatedConv1dReluBatchNorm(
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                scale=scale
            ),
            Conv1dReluBatchNorm(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            SEBlock(channels)
        )

    def forward(self, inputs):
        outputs = self.layers.forward(inputs)
        return outputs


class CnnTextDependentMultiScaleModel(nn.Module):
    def __init__(self):
        super(CnnTextDependentMultiScaleModel, self).__init__()
        pass


if __name__ == '__main__':
    pass
