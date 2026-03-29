import os
import unittest

import numpy as np
import sklearn.datasets
import torch
from sklearn.model_selection import train_test_split

if os.environ.get("ML_KATA_SOL"):
    import kata14.sol.train as sol
else:
    import kata14.train as sol


class TrainTestCase(unittest.TestCase):
    def test_xor(self):
        torch.manual_seed(0)
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0], dtype=np.int64)
        x = x.repeat(16, axis=0)
        y = y.repeat(16, axis=0)

        preds = sol.train(x, y, x, y, 2, 16)

        accuracy = (preds == y).mean()
        self.assertGreaterEqual(accuracy, 0.8)

    def test_mnist(self):
        data = sklearn.datasets.load_digits()
        train_inputs, valid_inputs, train_targets, valid_targets = train_test_split(
            data.data, data.target, random_state=0
        )
        torch.manual_seed(0)
        preds = sol.train(
            train_inputs, train_targets, valid_inputs, valid_targets, 10, 16
        )

        accuracy = (preds == valid_targets).mean()
        self.assertGreaterEqual(accuracy, 0.96)
        self.assertLessEqual(accuracy, 0.99)

    def test_mnist_anti_leakage(self):
        data = sklearn.datasets.load_digits()
        train_inputs, valid_inputs, train_targets, valid_targets = train_test_split(
            data.data, data.target, random_state=0
        )
        torch.manual_seed(0)
        preds = sol.train(
            train_inputs,
            train_targets,
            valid_inputs,
            np.zeros_like(valid_targets),
            10,
            16,
        )

        accuracy = (preds == valid_targets).mean()
        self.assertGreaterEqual(accuracy, 0.96)
        self.assertLessEqual(accuracy, 0.99)
