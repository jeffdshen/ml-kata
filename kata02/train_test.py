import os
import unittest

import numpy as np
import sklearn.datasets
import torch
from sklearn.model_selection import train_test_split

import kata02.sol.train as sol

if os.environ.get("ML_KATA_SOL"):
    import kata02.sol.train as sol
else:
    import kata02.train as sol


class TrainTestCase(unittest.TestCase):
    def test_mnist(self):
        data = sklearn.datasets.load_digits()
        train_inputs, valid_inputs, train_targets, valid_targets = train_test_split(
            data.data, data.target, random_state=0
        )
        torch.manual_seed(0)
        preds = sol.train(train_inputs, train_targets, valid_inputs, valid_targets, 16)

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
            train_inputs, train_targets, valid_inputs, np.zeros_like(valid_targets), 16
        )

        accuracy = (preds == valid_targets).mean()
        self.assertGreaterEqual(accuracy, 0.96)
        self.assertLessEqual(accuracy, 0.99)
