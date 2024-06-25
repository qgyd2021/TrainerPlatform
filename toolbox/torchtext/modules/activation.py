#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F


class Identity(nn.Module):

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


name2activation = {
    'linear': Identity,
    'relu': nn.ReLU,
}


if __name__ == '__main__':
    pass
