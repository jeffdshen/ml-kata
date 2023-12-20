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

        inputs = torch.from_numpy(inputs).to(torch.float32)
        targets = torch.from_numpy(targets)

        for i, (x, y) in enumerate(loader):
            torch.testing.assert_close(x, inputs[i * 8: (i + 1) * 8])
            torch.testing.assert_close(y, targets[i * 8: (i + 1) * 8])