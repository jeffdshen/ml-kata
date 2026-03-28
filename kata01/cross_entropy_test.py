import unittest

import torch
import torch.nn as nn

import os

if os.environ.get("ML_KATA_SOL"):
    import kata01.sol.cross_entropy as sol
else:
    import kata01.cross_entropy as sol


class CrossEntropyTestCase(unittest.TestCase):
    def check(self, cross_entropy, inputs, targets):
        outputs = cross_entropy(inputs, targets).sum() ** 2
        expected = {}
        expected["outputs"] = outputs.detach().clone()
        outputs.backward()

        expected["inputs.grad"] = inputs.grad.detach().clone()
        cross_entropy.zero_grad()
        inputs.grad.zero_()

        outputs = sol.CrossEntropyFunction.apply(inputs, targets).sum() ** 2
        outputs.backward()
        torch.testing.assert_close(outputs, expected["outputs"])
        torch.testing.assert_close(inputs.grad, expected["inputs.grad"])
        torch.autograd.gradcheck(
            sol.CrossEntropyFunction.apply, (inputs, targets), fast_mode=True
        )

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_dim_1(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        cross_entropy = nn.CrossEntropyLoss(reduction="none")
        inputs = torch.randn(64, requires_grad=True)
        targets = torch.randint(0, 64, tuple())
        self.check(cross_entropy, inputs, targets)

    def test_dim_2(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(1)
        cross_entropy = nn.CrossEntropyLoss(reduction="none")
        inputs = torch.randn(64, 64, requires_grad=True)
        targets = torch.randint(0, 64, (64,))
        self.check(cross_entropy, inputs, targets)
