#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://github.com/lawlict/ECAPA-TDNN
https://github.com/TaoRuijie/ECAPA-TDNN
https://arxiv.org/abs/2005.07143

"""
import math, torch, torchaudio
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


class AttentiveStatsPool(nn.Module):
    """
    计算每个频率段的均值,方差,以评价说话人特征.
    [8] K. Okabe, T. Koshinaka, and K. Shinoda, “Attentive statistics
    pooling for deep speaker embedding,” in Proc. Interspeech, 2018,
    pp. 2252–2256.
    """
    def __init__(self, inputs_dim: int, hidden_size: int):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.

        # equals W and b in the paper
        self.linear1 = nn.Conv1d(inputs_dim, hidden_size, kernel_size=1)

        # equals V and k in the paper
        self.linear2 = nn.Conv1d(hidden_size, inputs_dim, kernel_size=1)

    def forward(self, inputs):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        inputs = self.linear1(inputs)
        inputs = torch.tanh(inputs)
        inputs = self.linear2(inputs)

        alpha = torch.softmax(inputs, dim=2)

        mean = torch.sum(alpha * inputs, dim=2)

        residuals = torch.sum(alpha * inputs ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))

        return torch.cat([mean, std], dim=1)


class AAMSoftMax(nn.Module):
    """
    [6] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “ArcFace: Additive
    angular margin loss for deep face recognition,” in 2019 IEEE/CVF
    CVPR, 2019, pp. 4685–4694.

    [25] X. Xiang, S. Wang, H. Huang, Y. Qian, and K. Yu,
    “Margin matters: Towards more discriminative deep
    """
    def __init__(self, n_class, m, s):
        super(AAMSoftMax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        # prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        # return loss, prec1


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAugmentation(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class EcapaTdnnModel(nn.Module):
    """
    ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
    https://arxiv.org/abs/2005.07143
    """
    def __init__(self,
                 spec_dim=80,
                 hidden_size=512,
                 scale=8,
                 ):
        super(EcapaTdnnModel, self).__init__()
        self.layer1 = Conv1dReluBatchNorm(
            in_channels=spec_dim,
            out_channels=hidden_size,
            kernel_size=5,
            padding=2,
        )
        self.layer2 = SqueezeExcitationRes2Block(
            channels=hidden_size,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=scale,
        )
        self.layer3 = SqueezeExcitationRes2Block(
            channels=hidden_size,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=scale,
        )
        self.layer4 = SqueezeExcitationRes2Block(
            channels=hidden_size,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=scale,
        )

        self.conv = nn.Conv1d(
            in_channels=hidden_size * 3,
            out_channels=hidden_size * 3,
            kernel_size=1
        )

        self.pooling = AttentiveStatsPool(inputs_dim=hidden_size * 3, hidden_size=128)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 6)

    def forward(self, x):
        """
        :param x: shape = [batch_size, seq_length, spec_dim]
        :return:
        """
        x = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        sequence_output = torch.cat([out2, out3, out4], dim=1)

        # shape = [batch_size, channnels * 3, seq_length]
        sequence_output = F.relu(self.conv(sequence_output))

        # shape = [batch_size, channnels * 6]
        pooled_output = self.batch_norm(self.pooling(sequence_output))

        return pooled_output, sequence_output


class EcapaTdnnForSpectrumClassification(nn.Module):
    def __init__(self,
                 num_labels: int,
                 spec_dim: int = 80,
                 hidden_size: int = 512,
                 scale: int = 8,
                 ):
        super(EcapaTdnnForSpectrumClassification, self).__init__()
        self.ecapa_tdnn = EcapaTdnnModel(
            spec_dim=spec_dim,
            hidden_size=hidden_size,
            scale=scale,
        )
        self.linear = nn.Linear(hidden_size * 6, num_labels)
        self.batch_norm = nn.BatchNorm1d(num_labels)

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: shape = [batch_size, seq_length, spec_dim]
        :return:
        """
        pooled_output, _ = self.ecapa_tdnn.forward(inputs)
        pooled_output = self.linear(pooled_output)
        pooled_output = self.batch_norm(pooled_output)
        return pooled_output


class EcapaTdnnForWaveClassification(nn.Module):
    def __init__(self,
                 num_labels: int,
                 do_augmentation: bool = False,
                 sample_rate: int = 16000,
                 n_fft: int = 512,
                 win_length: int = 400,
                 hop_length: int = 160,
                 f_min: int = 20,
                 f_max: int = 7600,
                 window_fn: str = 'hamming',
                 n_mels: int = 80,
                 spec_dim: int = 80,
                 hidden_size: int = 512,
                 scale: int = 8,
                 ):
        super(EcapaTdnnForWaveClassification, self).__init__()
        self.do_augmentation = do_augmentation

        self.wave_to_mel_spectrogram = torch.nn.Sequential(
            PreEmphasis(),
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

        # Spec augmentation
        self.spec_augmentation = FbankAugmentation()

        self.ecapa_tdnn = EcapaTdnnForSpectrumClassification(
            num_labels=num_labels,
            spec_dim=spec_dim,
            hidden_size=hidden_size,
            scale=scale,
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
            if self.training and self.do_augmentation:
                x = self.spec_augmentation(x)

        x = x.transpose(1, 2)

        x = self.ecapa_tdnn.forward(x)
        return x


def demo1():
    model = EcapaTdnnModel()

    x = torch.ones(size=(2, 125, 80), dtype=torch.float32)
    outputs = model.forward(x)
    print(outputs.shape)
    return


def demo2():
    model = EcapaTdnnForSpectrumClassification(
        num_labels=2
    )

    x = torch.ones(size=(2, 125, 80), dtype=torch.float32)
    outputs = model.forward(x)
    print(outputs.shape)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
