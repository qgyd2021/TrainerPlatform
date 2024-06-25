#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict

import torch


def get_text_field_mask(text_field_tensors: torch.Tensor,
                        num_wrapping_dims: int = 0) -> torch.LongTensor:

    tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return ((character_tensor > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))


if __name__ == '__main__':
    pass
