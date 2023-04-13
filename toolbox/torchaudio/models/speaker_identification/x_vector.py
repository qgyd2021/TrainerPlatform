#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    """
    https://github.com/cvqluu/TDNN
    https://github.com/adelinocpp/x-vector_TDNN_python

    TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

    Affine transformation not applied globally to all frames but smaller windows with local context

    batch_norm: True to include batch normalisation after the non linearity

    Context size and dilation determine the frames selected
    (although context size is not really defined in the traditional sense)
    For example:
        context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
        context size 3 and dilation 2 is equivalent to [-2, 0, 2]
        context size 1 and dilation 1 is equivalent to [0]
    """
    def __init__(self,
                 input_dim: int = 23,
                 output_dim: int = 512,
                 context_size: int = 5,
                 stride: int = 1,
                 dilation: int = 1,
                 batch_norm: bool = True,
                 dropout: float = 0.2
                 ):
        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_size = context_size
        self.stride = stride
        self.dilation = dilation

        # n_values = input_dim * context_size
        self.linear = nn.Linear(input_dim * context_size, output_dim)
        self.activation = nn.ReLU()

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None

        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x):
        """
        :param x: shape=[batch_size, seq_length, input_dim]
        :return: shape=[batch_size, n_block, output_dim]
        """
        batch_size, seq_length1, input_dim = x.shape
        if input_dim != self.input_dim:
            raise AssertionError

        # shape=[batch_size, 1, seq_length, input_dim]
        x = torch.unsqueeze(x, dim=1)

        # shape=[batch_size, n_values, n_block]
        # shape=[batch_size, (context_size * input_dim), (seq_length - context_size + 1)]
        x = F.unfold(
            input=x,
            kernel_size=(self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1),
        )

        # shape=[batch_size, n_block, n_values]
        x = torch.transpose(x, dim0=1, dim1=2)

        # shape=[batch_size, n_block, output_dim]
        x = self.linear(x)
        x = self.activation(x)

        if self.dropout:
            x = self.dropout(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            # shape=[batch_size, output_dim, n_block]
            x = self.batch_norm(x)
            x = x.transpose(1, 2)

        return x


class XVectorModel(nn.Module):
    """
    https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
    """
    def __init__(self,
                 input_dim: int = 40,
                 hidden_size: int = 512,
                 dropout: float = 0.5
                 ):
        super(XVectorModel, self).__init__()
        self.tdnn = nn.Sequential(
            TDNN(
                input_dim=input_dim,
                output_dim=hidden_size,
                context_size=5,
                dilation=1,
                dropout=dropout
            ),
            TDNN(
                input_dim=hidden_size,
                output_dim=hidden_size,
                context_size=3,
                dilation=1,
                dropout=dropout
            ),
            TDNN(
                input_dim=hidden_size,
                output_dim=hidden_size,
                context_size=2,
                dilation=2,
                dropout=dropout
            ),
            TDNN(
                input_dim=hidden_size,
                output_dim=hidden_size,
                context_size=1,
                dilation=1,
                dropout=dropout
            ),
            TDNN(
                input_dim=hidden_size,
                output_dim=hidden_size,
                context_size=1,
                dilation=3,
                dropout=dropout
            )
        )

        self.project_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: shape=[batch_size, seq_length, spec_dim]
        :return:
        """
        x = self.tdnn(inputs)

        _, seq_length, _ = x.shape

        if seq_length > 1:
            mean = torch.mean(x, dim=1)
            std = torch.var(x, dim=1)
            stat_pooling = torch.cat(tensors=(mean, std), dim=1)
        else:
            mean = x.squeeze(dim=1)
            std = torch.zeros(mean.shape)
            stat_pooling = torch.cat(tensors=(mean, std), dim=1)

        x_vector = self.project_layers(stat_pooling)

        return x_vector


class XVectorForSpectrumClassification(nn.Module):
    def __init__(self,
                 input_dim: int = 40,
                 hidden_size: int = 512,
                 num_labels: int = 8,
                 dropout: float = 0.5
                 ):
        super(XVectorForSpectrumClassification, self).__init__()
        self.x_vector_model = XVectorModel(
            input_dim=input_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.output_project_layer = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: shape = [batch_size, seq_length, spec_dim]
        :return:
        """
        x_vector = self.x_vector_model.forward(inputs)
        logits = self.output_project_layer(x_vector)
        return logits


class XVectorForWaveClassification(nn.Module):
    def __init__(self,
                 num_labels: int,
                 sample_rate: int = 16000,
                 n_fft: int = 512,
                 win_length: int = 400,
                 hop_length: int = 160,
                 f_min: int = 20,
                 f_max: int = 7600,
                 window_fn: str = 'hamming',
                 n_mels: int = 80,
                 hidden_size: int = 512,
                 dropout: float = 0.5,
                 ):
        super(XVectorForWaveClassification, self).__init__()

        self.wave_to_mel_spectrogram = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                window_fn=torch.hamming_window if window_fn == 'hamming' else torch.hann_window,
                n_mels=n_mels,
            ),
        )
        self.x_vector_for_spectrum_classification = XVectorForSpectrumClassification(
            input_dim=n_mels,
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: shape = [batch_size, seq_length]
        :return:
        """
        x = inputs
        with torch.no_grad():
            # shape = [batch_size, spec_dim, seq_length]
            x = self.wave_to_mel_spectrogram(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)

        x = x.transpose(1, 2)

        x = self.x_vector_for_spectrum_classification.forward(x)
        return x


def demo1():
    tdnn = TDNN(input_dim=3, output_dim=3)

    inputs1 = torch.tensor(
        data=[
            [[1, 2, 3],
             [4, 5, 6],
             [1, 2, 3],
             [4, 5, 6],
             [1, 2, 3],
             [4, 5, 6]]
        ],
        dtype=torch.float32
    )
    # inputs1 = torch.ones(size=(1, 6, 3), dtype=torch.float32)
    inputs2 = torch.zeros(size=(1, 6, 3), dtype=torch.float32)
    inputs = torch.concat([inputs1, inputs2])

    outputs = tdnn.forward(inputs)
    print(outputs.shape)
    return


if __name__ == '__main__':
    demo1()
