import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAngularMarginLinear(nn.Module):
    """
    Alias: ArcFace, AAM-Softmax

    ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.07698

    参考代码:
    https://github.com/huangkeju/AAMSoftmax-OpenMax/blob/main/AAMSoftmax%2BOvA/metrics.py

    """
    @staticmethod
    def demo1():
        """
        角度与数值转换
        pi / 180 代表 1 度,
        pi / 180 = 0.01745
        """

        # 度数转数值
        degree = 10
        result = degree * math.pi / 180
        print(result)

        # 数值转数度
        radian = 0.2
        result = radian / (math.pi / 180)
        print(result)

        return

    @staticmethod
    def demo2():

        return

    def __init__(self,
                 hidden_size: int,
                 num_labels: int,
                 margin: float = 0.0,
                 scale: float = 1.0,
                 ):
        """
        :param hidden_size:
        :param num_labels:
        :param margin: 建议取值角度为 [10, 30], 对应的数值为 [0.1745, 0.5236]
        :param scale: 10.0
        """
        super(AdditiveAngularMarginLinear, self).__init__()
        self.margin = margin
        self.scale = scale
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_labels, hidden_size), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

        self.cos_margin = math.cos(self.margin)
        self.sin_margin = math.sin(self.margin)

        # sin(a-b) = sin(a)cos(b) - cos(a)sin(b)
        # sin(pi - a) = sin(a)

    def forward(self,
                inputs: torch.Tensor,
                label: torch.LongTensor = None
                ):
        """
        :param inputs: shape=[batch_size, ..., hidden_size]
        :param label:
        :return: logits
        """
        x = F.normalize(inputs)
        weight = F.normalize(self.weight)
        cosine = F.linear(x, weight)

        print('AdditiveAngularMarginLinear, self.training: {}'.format(self.training))
        if self.training:
            # sin^2  + cos^2 = 1
            sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))

            # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
            cosine_theta_margin = cosine * self.cos_margin - sine * self.sin_margin

            # when the `cosine > - self.cos_margin` there is enough space to add margin on theta.
            cosine_theta_margin = torch.where(cosine > - self.cos_margin, cosine_theta_margin, cosine - (self.margin * self.sin_margin))

            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)

            logits = torch.where(one_hot == 1, cosine_theta_margin, cosine)
            logits = logits * self.scale
        else:
            logits = cosine
        return logits


def demo1():
    AdditiveAngularMarginLinear.demo1()
    return


if __name__ == '__main__':
    demo1()
