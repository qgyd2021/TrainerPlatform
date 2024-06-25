#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data.dataset import Dataset


class DummyDataset(Dataset):
    def __init__(self, length: int = 10000):
        self.length = length
        data = np.random.random(size=(length,))
        labels = np.where(data > 0.5, 1, 0)
        self.samples = list(zip(data, labels))

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    pass
