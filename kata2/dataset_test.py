import unittest

import numpy as np

import torch
from torch.utils.data import DataLoader

import kata2.dataset as sol

class NumpyDatasetTestCase(unittest.TestCase):
    def test_dataset(self):
        np.random.seed(0)
        inputs = np.random.randn(100, 64)
        targets = np.random.randint(0, 10, (100,))
        dataset = sol.NumpyDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=8)

        for i, (x, y) in enumerate(loader):
            torch.testing.assert_allclose(x, inputs[i * 8: (i + 1) * 8])
            torch.testing.assert_allclose(y, targets[i * 8: (i + 1) * 8])