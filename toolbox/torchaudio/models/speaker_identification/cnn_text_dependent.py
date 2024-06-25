#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List, Union, Tuple

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


name2activation = {
    'relu': nn.ReLU,
}


class Conv1dBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: Tuple[int, int],
                 padding: str = 0,
                 dilation: int = 1,
                 batch_norm: bool = False,
                 activation: str = None,
                 dropout: float = None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(in_channels)
        else:
            self.batch_norm = None

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size,),
            stride=stride,
            padding=padding,
            dilation=(dilation,),
        )

        if activation is None:
            self.activation = None
        else:
            self.activation = name2activation[activation]()

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # x: [batch_size, seq_length, spec_dim]
        x = torch.transpose(x, dim0=-1, dim1=-2)

        # x: [batch_size, spec_dim, seq_length]
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.conv(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = torch.transpose(x, dim0=-1, dim1=-2)
        # x: [batch_size, seq_length, spec_dim]
        return x


class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Tuple[int, int],
                 padding: str = 0,
                 dilation: int = 1,
                 batch_norm: bool = False,
                 activation: str = None,
                 dropout: float = None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: Tuple[int, int] = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(in_channels)
        else:
            self.batch_norm = None

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding,),
            dilation=(dilation,),
        )

        if activation is None:
            self.activation = None
        else:
            self.activation = name2activation[activation]()

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x):

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.conv(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class FeedForward(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, List[int]],
                 activations: Union[str, List[str]],
                 dropout: Union[float, List[float]] = 0.0) -> None:

        super(FeedForward, self).__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore
        if len(hidden_dims) != num_layers:
            raise AssertionError("len(hidden_dims) (%d) != num_layers (%d)" %
                                 (len(hidden_dims), num_layers))
        if len(activations) != num_layers:
            raise AssertionError("len(activations) (%d) != num_layers (%d)" %
                                 (len(activations), num_layers))
        if len(dropout) != num_layers:
            raise AssertionError("len(dropout) (%d) != num_layers (%d)" %
                                 (len(dropout), num_layers))
        self._activations = torch.nn.ModuleList([name2activation[activation]() for activation in activations])

        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self.output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            output = dropout(activation(layer(output)))
        return output


class CnnTextDependentModel(nn.Module):
    """
    https://arxiv.org/abs/1703.05390
    """

    @staticmethod
    def demo1():
        inputs = torch.ones(size=(2, 125, 80), dtype=torch.float32)
        cnn = CnnTextDependentModel(
            conv1d_block_param_list=[
                {
                    'batch_norm': True,
                    'in_channels': 80,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
                {
                    # 'batch_norm': True,
                    'in_channels': 16,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
                {
                    # 'batch_norm': True,
                    'in_channels': 16,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
            ],
            feedforward_param={
                'input_dim': 16,
                'num_layers': 2,
                'hidden_dims': 32,
                'activations': 'relu',
                'dropout': 0.1,
            }
        )

        outputs = cnn.forward(inputs)
        print(outputs.shape)

        return

    def __init__(self,
                 feedforward_param: dict,
                 conv1d_block_param_list: List[dict] = None,
                 conv2d_block_param_list: List[dict] = None,
                 ):
        super().__init__()
        self.conv1d_block_list = None
        if conv1d_block_param_list is not None:
            self.conv1d_block_list = nn.ModuleList(modules=[
                Conv1dBlock(**conv1d_block_param)
                for conv1d_block_param in conv1d_block_param_list
            ])

        self.conv2d_block_list = None
        if conv2d_block_param_list is not None:
            self.conv2d_block_list = nn.ModuleList(modules=[
                Conv2dBlock(**conv2d_block_param)
                for conv2d_block_param in conv2d_block_param_list
            ])

        self.feedforward = FeedForward(**feedforward_param)

    def get_output_dim(self):
        return self.feedforward.output_dim

    def forward(self,
                inputs: torch.Tensor,
                ):
        x = inputs
        # x: [batch_size, spec_dim, seq_length]

        if self.conv1d_block_list is not None:
            for conv1d_block in self.conv1d_block_list:
                x = conv1d_block(x)

        if self.conv2d_block_list is not None:
            x = torch.unsqueeze(x, dim=1)
            # x: [batch_size, channel, spec_dim, seq_length]
            for conv2d_block in self.conv2d_block_list:
                x = conv2d_block(x)

            # x: [batch_size, channel, spec_dim, seq_length]
            x = torch.transpose(x, dim0=1, dim1=3)
            # x: [batch_size, seq_length, spec_dim, channel]

            batch_size, seq_length, spec_dim, channel = x.shape
            x = torch.reshape(x, shape=(batch_size, seq_length, -1))

        # x: [batch_size, seq_length, spec_dim]
        x = self.feedforward(x)
        return x


class CnnTextDependentSpectrumClassifier(nn.Module):
    @staticmethod
    def demo1():
        inputs = torch.ones(size=(2, 125, 80), dtype=torch.float32)
        cnn = CnnTextDependentSpectrumClassifier(
            num_labels=2,
            conv1d_block_param_list=[
                {
                    'batch_norm': True,
                    'in_channels': 80,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
                {
                    # 'batch_norm': True,
                    'in_channels': 16,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
                {
                    # 'batch_norm': True,
                    'in_channels': 16,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
            ],
            feedforward_param={
                'input_dim': 16,
                'num_layers': 2,
                'hidden_dims': 32,
                'activations': 'relu',
                'dropout': 0.1,
            }
        )

        outputs = cnn.forward(inputs)
        print(outputs.shape)

        return

    def __init__(self,
                 num_labels: int,
                 feedforward_param: dict,
                 conv1d_block_param_list: List[dict] = None,
                 conv2d_block_param_list: List[dict] = None,
                 ):
        super().__init__()
        self.model = CnnTextDependentModel(
            feedforward_param=feedforward_param,
            conv1d_block_param_list=conv1d_block_param_list,
            conv2d_block_param_list=conv2d_block_param_list,
        )
        self.output_project_layer = nn.Linear(self.model.get_output_dim(), num_labels)

    def forward(self, inputs: torch.Tensor):
        x = inputs
        # shape = [batch_size, spec_dim, seq_length]
        x = self.model.forward(x)
        # shape = [batch_size, spec_dim, seq_length]
        x = torch.mean(x, dim=1)
        logits = self.output_project_layer.forward(x)
        return logits


class CnnTextDependentWaveClassifier(nn.Module):
    @staticmethod
    def demo1():
        inputs = torch.ones(size=(2, 16000), dtype=torch.float32)
        cnn = CnnTextDependentWaveClassifier(
            num_labels=2,
            conv1d_block_param_list=[
                {
                    'batch_norm': True,
                    'in_channels': 80,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
                {
                    # 'batch_norm': True,
                    'in_channels': 16,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
                {
                    # 'batch_norm': True,
                    'in_channels': 16,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 3,
                    # 'padding': 'same',
                    'activation': 'relu',
                    'dropout': 0.1,
                },
            ],
            feedforward_param={
                'input_dim': 16,
                'num_layers': 2,
                'hidden_dims': 32,
                'activations': 'relu',
                'dropout': 0.1,
            },
            mel_spectrogram_param={
                'sample_rate': 8000,
                'n_fft': 512,
                'win_length': 200,
                'hop_length': 80,
                'f_min': 10,
                'f_max': 3800,
                'window_fn': 'hamming',
                'n_mels': 80,
            }
        )

        outputs = cnn.forward(inputs)
        print(outputs.shape)

        return

    def __init__(self,
                 num_labels: int,
                 feedforward_param: dict,
                 mel_spectrogram_param: dict,
                 conv1d_block_param_list: List[dict] = None,
                 conv2d_block_param_list: List[dict] = None,
                 ):
        super().__init__()

        self.wave_to_mel_spectrogram = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=mel_spectrogram_param['sample_rate'],
                n_fft=mel_spectrogram_param['n_fft'],
                win_length=mel_spectrogram_param['win_length'],
                hop_length=mel_spectrogram_param['hop_length'],
                f_min=mel_spectrogram_param['f_min'],
                f_max=mel_spectrogram_param['f_max'],
                window_fn=torch.hamming_window if mel_spectrogram_param['window_fn'] == 'hamming' else torch.hann_window,
                n_mels=mel_spectrogram_param['n_mels'],
            ),
        )

        self.model = CnnTextDependentModel(
            feedforward_param=feedforward_param,
            conv1d_block_param_list=conv1d_block_param_list,
            conv2d_block_param_list=conv2d_block_param_list,
        )
        self.output_project_layer = nn.Linear(self.model.get_output_dim(), num_labels)

    def forward(self, inputs: torch.Tensor):
        x = inputs

        with torch.no_grad():
            # shape = [batch_size, spec_dim, seq_length]
            x = self.wave_to_mel_spectrogram(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)

        x = x.transpose(1, 2)

        # shape = [batch_size, seq_length, spec_dim]
        x = self.model.forward(x)

        x = torch.mean(x, dim=1)
        logits = self.output_project_layer.forward(x)
        return logits


def demo1():
    CnnTextDependentWaveClassifier.demo1()
    return


if __name__ == '__main__':
    demo1()
