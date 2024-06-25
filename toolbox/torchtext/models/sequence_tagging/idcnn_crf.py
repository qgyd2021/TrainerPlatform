#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List

import torch
import torch.nn as nn


class IDCNNBlock(nn.Module):
    """
    linear
    conv1d
    conv1d
    dilation_conv1d
    relu
    layer_norm
    """
    def __init__(self,
                 inputs_dim: int,
                 hidden_size: int,
                 kernel_size: int,
                 dilation: int = 2,
                 ):
        super(IDCNNBlock, self).__init__()
        self.linear = nn.Linear(inputs_dim, hidden_size)
        self.conv0 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            dilation=1,
            padding=(kernel_size - 1) // 2,
        )
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            dilation=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dilated_conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) // 2 * dilation,
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: shape=[batch_size, max_seq_length, inputs_dim]
        :return:
        """
        x = self.linear.forward(inputs)
        x = torch.transpose(x, dim0=1, dim1=2)
        x = self.conv0.forward(x)
        x = self.conv1.forward(x)
        x = self.dilated_conv2.forward(x)
        x = torch.transpose(x, dim0=1, dim1=2)
        x = self.relu(x)
        x = self.layer_norm(x)
        return x


class IDCNN(nn.Module):

    """
    https://arxiv.org/abs/1702.02098

    官方代码 (tensorflow):
    https://github.com/iesl/dilated-cnn-ner

    参考代码:
    https://github.com/ZeroE04/IDCNN-pytorch
    """

    @staticmethod
    def demo1():
        embedding = nn.Embedding(10, 5)
        id_cnn = IDCNN(
            inputs_dim=5,
            hidden_size=4,
            n_blocks_params=[
                {
                    'kernel_size': 3,
                    'dilation': 2,
                },
                {
                    'kernel_size': 3,
                    'dilation': 2,
                },
                {
                    'kernel_size': 3,
                    'dilation': 2,
                },
                {
                    'kernel_size': 3,
                    'dilation': 2,
                },
            ]
        )

        token_ids = torch.tensor([[1, 2, 2, 3, 1, 0]], dtype=torch.long)
        text_embeded = embedding.forward(token_ids)
        print(text_embeded.shape)
        outputs = id_cnn.forward(text_embeded)
        print(outputs.shape)
        return

    def __init__(self,
                 inputs_dim: int,
                 hidden_size: int,
                 n_blocks_params: List[dict],
                 ):
        super(IDCNN, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList(modules=[
            IDCNNBlock(
                inputs_dim=inputs_dim if idx == 0 else hidden_size,
                hidden_size=hidden_size,
                kernel_size=block_params['kernel_size'],
                dilation=block_params['dilation'],
            ) for idx, block_params in enumerate(n_blocks_params)
        ])

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: shape=[batch_size, max_seq_length, inputs_dim]
        :return:
        """
        x = inputs
        for block in self.blocks:
            x = block.forward(x)
        return x


def demo1():
    IDCNN.demo1()
    return


if __name__ == '__main__':
    demo1()
