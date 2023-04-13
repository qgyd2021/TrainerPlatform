#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
from torchaudio.models.tacotron2 import Tacotron2, _Encoder
from typing import List, Optional, Tuple

from toolbox.torch.modules.highway import Highway


class Tacotron2Encoder(nn.Module):

    @staticmethod
    def demo1():
        encoder = Tacotron2Encoder()

        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        token_lengths = torch.tensor([5])
        result = encoder.forward(tokens, token_lengths)
        print(result.shape)
        return

    def __init__(self,
                 vocab_size: int = 148,
                 encoder_embedding_dim: int = 512,
                 encoder_n_convolution: int = 3,
                 encoder_kernel_size: int = 5,
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, encoder_embedding_dim)
        std = math.sqrt(2.0 / (vocab_size + encoder_embedding_dim))
        val = math.sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = _Encoder(
            encoder_embedding_dim,
            encoder_n_convolution,
            encoder_kernel_size
        )

        self._output_dim = encoder_embedding_dim

    def get_output_dim(self):
        return self._output_dim

    def forward(self,
                tokens: torch.Tensor,
                token_lengths: torch.Tensor,
                ):
        embedded_inputs = self.embedding(tokens).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, token_lengths)
        return encoder_outputs

    def infer(self, tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_batch, max_length = tokens.shape
        if lengths is None:
            lengths = torch.tensor([max_length]).expand(n_batch).to(tokens.device, tokens.dtype)

        assert lengths is not None  # For TorchScript compiler

        embedded_inputs = self.embedding(tokens).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, lengths)
        return encoder_outputs


class BatchNormConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 activation=None
                 ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class BatchNormConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 activation=None
                 ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv2d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class ContentEncoder(nn.Module):

    @staticmethod
    def demo1():
        encoder = ContentEncoder()

        inputs = torch.rand(size=(1, 125, 80))
        input_lengths = torch.tensor([5])
        result = encoder.forward(inputs, input_lengths)
        print(result.shape)
        return

    def __init__(self,
                 in_channels: int = 80,
                 kernel_size: int = 17,
                 filter_sizes: List[int] = None,
                 gru_hidden_size: int = 256,
                 ):
        super().__init__()
        filter_sizes = filter_sizes or [512, 512, 512]

        filters = [in_channels] + filter_sizes

        self.convs = nn.Sequential(*[
            BatchNormConv1d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            )
            for i in range(len(filter_sizes))
        ])

        self.gru = nn.GRU(
            input_size=filter_sizes[-1],
            hidden_size=gru_hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=gru_hidden_size * 2,
            hidden_size=gru_hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, inputs, input_lengths):
        x = self.convs(inputs.transpose(1, 2)).transpose(1, 2)

        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x, _ = self.gru2(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x


class BidirectionalAttention(nn.Module):
    """
    https://arxiv.org/abs/2203.16838

    """
    @staticmethod
    def demo1():
        attention = BidirectionalAttention(
            k1_dim=128,
            k2_dim=80,
            v1_dim=128,
            v2_dim=80,
            attention_dim=128,
        )

        k1 = torch.rand(size=(3, 3, 128))
        k2 = torch.rand(size=(3, 5, 80))

        k1_lengths = torch.tensor(data=[2, 3, 1], dtype=torch.long)
        k2_lengths = torch.tensor(data=[4, 5, 2], dtype=torch.long)

        o1, o2, w1, w2, score = attention.forward(
            k1=k1, k2=k2,
            v1=k1, v2=k2,
            k1_lengths=k1_lengths,
            k2_lengths=k2_lengths
        )
        print(o1.shape)
        print(o2.shape)
        return

    def __init__(self,
                 k1_dim: int,
                 k2_dim: int,
                 v1_dim: int,
                 v2_dim: int,
                 attention_dim: int,
                 ):
        super().__init__()
        self.k1_layer = nn.Linear(k1_dim, attention_dim)
        self.k2_layer = nn.Linear(k2_dim, attention_dim)
        self.score_layer = nn.Linear(attention_dim, 1)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

    def forward(self, k1, k2, v1, v2, k1_lengths=None, k2_lengths=None):
        batch_size1, max_seq_length1, _ = k1.shape
        batch_size2, max_seq_length2, _ = k2.shape
        if batch_size1 != batch_size2:
            raise AssertionError

        batch_size = batch_size1

        if k1_lengths is None:
            k1_lengths = torch.ones(size=(batch_size,), dtype=torch.long).data.fill_(max_seq_length1)
        if k2_lengths is None:
            k2_lengths = torch.ones(size=(batch_size,), dtype=torch.long).data.fill_(max_seq_length2)

        # [batch_size, seq_length1]
        mask1 = torch.arange(max_seq_length1).unsqueeze(dim=0).repeat(batch_size, 1)
        mask1 = mask1 < k1_lengths.unsqueeze(dim=1)
        mask1 = mask1.long()

        # [batch_size, seq_length2]
        mask2 = torch.arange(max_seq_length2).unsqueeze(dim=0).repeat(batch_size, 1)
        mask2 = mask2 < k2_lengths.unsqueeze(dim=1)
        mask2 = mask2.long()

        # [batch_size, seq_length1, seq_length2]
        mask1_ = mask1.unsqueeze(dim=2)
        mask2_ = mask2.unsqueeze(dim=1)
        attention_mask = mask1_ + mask2_
        attention_mask = attention_mask == 1

        # [batch_size, seq_length1, dim]
        k1 = self.k1_layer(k1)
        # [batch_size, seq_length2, dim]
        k2 = self.k2_layer(k2)

        # [batch_size, seq_length1, seq_length2]
        score = torch.bmm(k1, k2.transpose(1, 2))
        score = score.masked_fill_(attention_mask, -float('inf'))

        # [batch_size, seq_length2, seq_length1]
        w1 = self.softmax1(score.transpose(1, 2))
        # [batch_size, seq_length1, seq_length2]
        w2 = self.softmax2(score)

        # [batch_size, seq_length2, k1_dim]
        o1 = torch.bmm(w1, v1)
        # [batch_size, seq_length1, k2_dim]
        o2 = torch.bmm(w2, v2)

        o1 = o1 * mask2.unsqueeze(dim=-1)
        o2 = o2 * mask1.unsqueeze(dim=-1)

        return o1, o2, w1, w2, score


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('div_term', div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, custom_position=None):
        if custom_position is not None:
            custom_position = custom_position.repeat(self.div_term.shape[0], 1, 1).permute(1, 2, 0)
            pe = torch.zeros(custom_position.shape[:-1] + (self.d_model, ))
            pe[:, :, 0::2] = torch.sin(custom_position * self.div_term)
            pe[:, :, 1::2] = torch.cos(custom_position * self.div_term)
            x = x + pe.to(x.device)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 ):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self,
                inputs: torch.Tensor,
                input_lengths: torch.LongTensor,
                ):
        x = nn.utils.rnn.pack_padded_sequence(
            input=inputs,
            lengths=input_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=x,
            batch_first=True
        )
        x = self.linear(x)
        return x


class Aligner(nn.Module):

    def __init__(self,
                 max_frames: int = 4500,
                 in_channels: int = 6,
                 kernel_size: int = 17,
                 filter_sizes: List[int] = None,
                 ):
        super().__init__()
        self.max_frames = max_frames
        filter_sizes = filter_sizes or [32, 32, 32]
        filters = [in_channels] + filter_sizes

        padding = (kernel_size - 1) // 2
        self.convs = nn.Sequential(
            *[
                BatchNormConv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
                for i in range(len(filter_sizes))
            ]
        )

        self.linear = nn.Linear(in_features=filters[-1], out_features=2)
        self.softmax = nn.Softmax(dim=-1)

    def stack_attention(self,
                        w1: torch.Tensor,
                        w2: torch.Tensor,
                        ):
        """
        :param w1: shape=[batch_size, max_seq_length2, max_seq_length1]
        :param w2: shape=[batch_size, max_seq_length1, max_seq_length2]
        :return:
        """
        w1 = w1.permute(0, 2, 1)

        accumulated_w1 = torch.cumsum(w1, -1)
        accumulated_w2 = torch.cumsum(w2, -1)

        accumulated_w1_backward = torch.cumsum(w1.flip(-1), -1).flip(-1)
        accumulated_w2_backward = torch.cumsum(w2.flip(-1), -1).flip(-1)
        x = torch.stack(
            [w1, w2, accumulated_w1, accumulated_w2, accumulated_w1_backward, accumulated_w2_backward],
            dim=-1
        )

        # [batch_size, 6, max_seq_length1, max_seq_length2]
        x = x.permute(0, 3, 1, 2)
        x = self.convs(x)
        # [batch_size, max_seq_length1, max_seq_length2, n_filters]
        x = x.permute(0, 2, 3, 1)
        return x

    def forward(self,
                w1: torch.Tensor,
                w2: torch.Tensor,
                token_lengths: torch.LongTensor,
                spectrum_lengths: torch.LongTensor
                ):
        """
        :param w1: shape=[batch_size, max_seq_length2, max_seq_length1]
        :param w2: shape=[batch_size, max_seq_length1, max_seq_length2]
        :param token_lengths: shape=[batch_size, max_seq_length1]
        :param spectrum_lengths: shape=[batch_size, max_seq_length2]
        :return: boundaries
        """

        # [batch_size, max_seq_length1, max_seq_length2, n_filters]
        x = self.stack_attention(w1, w2)

        # [batch_size, max_seq_length1, max_seq_length2, 2]
        x = self.linear(x)
        x = torch.sigmoid(x)

        x = x.transpose(-1, -2)

        x = torch.cumsum(x, dim=-1)

        # [batch_size, max_seq_length1, 2, max_seq_length2]
        boundaries = torch.tanh(x)
        return boundaries


class NeuFA(nn.Module):
    """
    https://arxiv.org/abs/2203.16838
    https://github.com/thuhcsi/NeuFA
    """
    def __init__(self,
                 vocab_size: int = 10,
                 n_mels: int = 80,
                 text_encoder_hidden_size: int = 512,
                 speech_encoder_hidden_size: int = 512,
                 attention_dim: int = 512,

                 text_decoder_input_size: int = 512,
                 text_decoder_hidden_size: int = 512,
                 text_decoder_output_size: int = 512,

                 speech_decoder_input_size: int = 512,
                 speech_decoder_hidden_size: int = 512,
                 speech_decoder_output_size: int = 512,
                 ):
        super(NeuFA, self).__init__()
        self.text_encoder = Tacotron2Encoder(
            vocab_size=vocab_size,
            encoder_embedding_dim=text_encoder_hidden_size
        )
        self.speech_encoder = ContentEncoder(
            in_channels=n_mels,
            gru_hidden_size=speech_encoder_hidden_size // 2
        )

        self.positional_encoding_text = PositionalEncoding(
            embedding_dim=text_encoder_hidden_size
        )
        self.positional_encoding_speech = PositionalEncoding(
            embedding_dim=speech_encoder_hidden_size
        )

        self.attention = BidirectionalAttention(
            k1_dim=text_encoder_hidden_size,
            k2_dim=speech_encoder_hidden_size,
            v1_dim=text_encoder_hidden_size,
            v2_dim=speech_encoder_hidden_size,
            attention_dim=attention_dim,
        )

        self.text_decoder = Decoder(
            input_size=text_decoder_input_size,
            hidden_size=text_decoder_hidden_size,
            output_size=text_decoder_output_size,
        )
        self.speech_decoder = Decoder(
            input_size=speech_decoder_input_size,
            hidden_size=speech_decoder_hidden_size,
            output_size=speech_decoder_output_size,
        )

        self.aligner = Aligner(
            max_frames=4500,
            in_channels=6,
            kernel_size=17,
            filter_sizes=[32, 32, 32],
        )

    def forward(self,
                tokens: torch.Tensor,
                token_lengths: torch.LongTensor,
                spectrums: torch.Tensor,
                spectrum_lengths: torch.LongTensor,
                ):
        """
        :param tokens: shape=[batch_size, max_seq_length1]
        :param token_lengths: shape=[batch_size,]
        :param spectrums: shape=[batch_size, max_seq_length2]
        :param spectrum_lengths: shape=[batch_size,]
        :return:
        """
        if token_lengths is None:
            token_lengths = tokens.not_equal(0).sum(dim=-1)
        token_encoded = self.text_encoder.forward(tokens, token_lengths)

        spectrum_encoded = self.speech_encoder.forward(spectrums, spectrum_lengths)

        token_encoded = self.positional_encoding_text(token_encoded)
        spectrum_encoded = self.positional_encoding_speech(spectrum_encoded)

        token_to_spectrum, spectrum_to_token, w1, w2, _ = self.attention.forward(
            k1=token_encoded,
            k2=spectrum_encoded,
            v1=token_encoded,
            v2=spectrum_encoded,
            k1_lengths=token_lengths,
            k2_lengths=spectrum_lengths,
        )
        # w1: [batch_size, max_seq_length2, max_seq_length1]
        w1 = self.sanitize_weight1(w1, k1_lengths=token_lengths, k2_lengths=spectrum_lengths)
        # w2: [batch_size, max_seq_length1, max_seq_length2]
        w2 = self.sanitize_weight2(w2, k1_lengths=token_lengths, k2_lengths=spectrum_lengths)

        # [batch_size, max_seq_length1, text_decoder_output_size]
        spectrum_to_token = self.text_decoder.forward(
            spectrum_to_token, token_lengths
        )
        spectrum_to_token = self.sanitize_spectrum_to_token(spectrum_to_token, token_lengths)

        # [batch_size, max_seq_length2, speech_decoder_output_size]
        token_to_spectrum = self.speech_decoder.forward(
            token_to_spectrum, spectrum_lengths
        )
        token_to_spectrum = self.sanitize_token_to_spectrum(token_to_spectrum, spectrum_lengths)

        # [batch_size, max_seq_length1, 2, max_seq_length2]
        boundaries = self.aligner.forward(
            w1=w1,
            w2=w2,
            token_lengths=token_lengths,
            spectrum_lengths=spectrum_lengths,
        )

        # result = NeuFALoss.extract_boundary(boundaries)
        # print(result)
        return spectrum_to_token, token_to_spectrum, w1, w2, boundaries

    def sanitize_weight1(self,
                         w1: torch.Tensor,
                         k1_lengths: torch.LongTensor,
                         k2_lengths: torch.LongTensor
                         ):
        w1_ = list()
        for w, l1, l2 in zip(w1, k1_lengths, k2_lengths):
            height, width = w.shape
            w = w[:l2, :l1]
            w = F.pad(input=w, pad=(0, width - l1, 0, height - l2), mode='constant', value=0.0)
            w1_.append(w)
        w1 = torch.stack(w1_, dim=0)
        return w1

    def sanitize_weight2(self,
                         w2: torch.Tensor,
                         k1_lengths: torch.LongTensor,
                         k2_lengths: torch.LongTensor
                         ):
        w2_ = list()
        for w, l1, l2 in zip(w2, k1_lengths, k2_lengths):
            height, width = w.shape
            w = w[:l1, :l2]
            w = F.pad(input=w, pad=(0, width - l2, 0, height - l1), mode='constant', value=0.0)
            w2_.append(w)
        w2 = torch.stack(w2_, dim=0)
        return w2

    def sanitize_spectrum_to_token(self,
                                   spectrum_to_token: torch.Tensor,
                                   token_lengths: torch.LongTensor,
                                   ):
        spectrum_to_token_ = list()
        for s_to_t, l in zip(spectrum_to_token, token_lengths):
            height, width = s_to_t.shape
            s_to_t = s_to_t[:l, :]
            s_to_t = F.pad(input=s_to_t, pad=(0, 0, 0, height - l), mode='constant', value=0.0)
            spectrum_to_token_.append(s_to_t)

        spectrum_to_token = torch.stack(spectrum_to_token_, dim=0)
        return spectrum_to_token

    def sanitize_token_to_spectrum(self,
                                   token_to_spectrum: torch.Tensor,
                                   spectrum_lengths: torch.LongTensor,
                                   ):
        token_to_spectrum_ = list()
        for t_to_s, l in zip(token_to_spectrum, spectrum_lengths):
            height, width = t_to_s.shape
            t_to_s = t_to_s[:l, :]
            t_to_s = F.pad(input=t_to_s, pad=(0, 0, 0, height - l), mode='constant', value=0.0)
            token_to_spectrum_.append(t_to_s)

        token_to_spectrum = torch.stack(token_to_spectrum_, dim=0)
        return token_to_spectrum

    def sanitize_boundaries(self,
                            boundaries: torch.Tensor,
                            ):

        return boundaries


class NeuFALoss(nn.Module):
    def __init__(self,
                 token_loss_weight: float = 0.1,
                 spectrum_loss_weight: float = 1.0,
                 attention_loss_weight: float = 1e-3,
                 attention_loss_alpha: float = 0.5,
                 boundary_loss_weight: float = 100,
                 ):
        super(NeuFALoss, self).__init__()
        self.token_loss_weight = token_loss_weight
        self.spectrum_loss_weight = spectrum_loss_weight
        self.attention_loss_weight = attention_loss_weight
        self.attention_loss_alpha = attention_loss_alpha
        self.boundary_loss_weight = boundary_loss_weight

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self,
                spectrum_logits: torch.Tensor,
                spectrum_targets: torch.Tensor,

                tokens_logits: torch.Tensor,
                tokens_targets: torch.Tensor,

                w1: torch.Tensor,
                w2: torch.Tensor,
                ):
        """
        :param spectrum_logits: [batch_size, max_seq_length2, speech_decoder_output_size]
        :param spectrum_targets: [batch_size, max_seq_length2, n_mels]
        :param spectrum_lengths:
        :param tokens_logits: [batch_size, max_seq_length1, text_decoder_output_size]
        :param tokens_targets: [batch_size, max_seq_length1, vocab_size]
        :param w1: [batch_size, max_seq_length2, max_seq_length1]
        :param w2: [batch_size, max_seq_length1, max_seq_length2]
        :return:
        """
        spectrum_loss = self.mse.forward(spectrum_logits, spectrum_targets)
        spectrum_loss = self.spectrum_loss_weight * spectrum_loss
        # print(spectrum_loss)

        tokens_logits = tokens_logits.permute(dims=[0, 2, 1])
        token_loss = self.cross_entropy.forward(tokens_logits, tokens_targets)
        token_loss = self.token_loss_weight * token_loss
        # print(token_loss)

        # attention loss
        attention_loss = []
        for _w1, _w2 in zip(w1, w2):
            w = torch.maximum(_w1.T, _w2)
            a = torch.linspace(1e-6, 1, w.shape[0], device=w.device).repeat(w.shape[1], 1).T
            b = torch.linspace(1e-6, 1, w.shape[1], device=w.device).repeat(w.shape[0], 1)
            r1 = torch.maximum((a / b), (b / a))
            r2 = torch.maximum(a.flip(1) / b.flip(0), b.flip(0) / a.flip(1))
            r = torch.maximum(r1, r2) - 1
            r = torch.tanh(self.attention_loss_alpha * r)
            attention_loss.append(torch.mean(w * r.detach()))
        attention_loss = torch.stack(attention_loss)
        attention_loss = torch.mean(attention_loss)
        attention_loss = self.attention_loss_weight * attention_loss
        # print(attention_loss)

        # boundary mae loss
        # pass

        loss = spectrum_loss + token_loss + attention_loss
        return loss

    @staticmethod
    def extract_boundary(boundaries, threshold=0.5):
        result = []
        for boundary in boundaries:
            result.append([])
            result[-1].append(torch.FloatTensor([i[i<threshold].shape[0] / 100 for i in boundary[:,0,:]]))
            result[-1].append(torch.FloatTensor([i[i<threshold].shape[0] / 100 for i in boundary[:,1,:]]))
            result[-1] = torch.stack(result[-1], dim=-1).to(boundaries[0].device)
        return result


def demo1():
    vocab_size = 10
    n_mels = 20
    neufa = NeuFA(
        vocab_size=vocab_size,
        n_mels=n_mels,
        text_encoder_hidden_size=128,
        speech_encoder_hidden_size=128,
        attention_dim=128,

        text_decoder_input_size=128,
        text_decoder_hidden_size=128,
        text_decoder_output_size=vocab_size,

        speech_decoder_input_size=128,
        speech_decoder_hidden_size=128,
        speech_decoder_output_size=n_mels,
    )

    neufa_loss = NeuFALoss()

    tokens = torch.tensor([
        [1, 2, 3, 4, 5],
        [2, 3, 2, 1, 0],
        [5, 4, 3, 0, 0]
    ], dtype=torch.long)

    spectrums = torch.rand(size=(3, 7, 20))

    token_lengths: torch.LongTensor = torch.tensor(data=[5, 4, 3], dtype=torch.long)
    spectrum_lengths: torch.LongTensor = torch.tensor(data=[4, 7, 5], dtype=torch.long)

    spectrum_to_token, token_to_spectrum, w1, w2, boundaries = neufa.forward(
        tokens=tokens,
        token_lengths=token_lengths,
        spectrums=spectrums,
        spectrum_lengths=spectrum_lengths,
    )

    neufa_loss.forward(
        token_to_spectrum, token_to_spectrum,
        spectrum_to_token, tokens,
        w1, w2,
    )
    return


if __name__ == '__main__':
    demo1()
