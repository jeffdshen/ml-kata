import unittest
import torch
import torch.nn as nn

import numpy as np
import kata1.ce as sol


class CeTestCase(unittest.TestCase):
    def check(self, ce, inputs, targets):
        outputs = ce(inputs, targets).sum() ** 2
        expected = {}
        expected["outputs"] = outputs.detach().clone()
        outputs.backward()

        expected["inputs.grad"] = inputs.grad.detach().clone()
        ce.zero_grad()
        inputs.grad.zero_()

        outputs = sol.CeFunction.apply(inputs, targets).sum() ** 2
        outputs.backward()
        torch.testing.assert_close(outputs, expected["outputs"])
        torch.testing.assert_close(inputs.grad, expected["inputs.grad"])
        torch.autograd.gradcheck(
            sol.CeFunction.apply, (inputs, targets), fast_mode=True
        )

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_dim_1(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        ce = nn.CrossEntropyLoss(reduction="none")
        inputs = torch.randn(64, requires_grad=True)
        targets = torch.randint(0, 64, tuple())
        self.check(ce, inputs, targets)

    def test_dim_2(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        ce = nn.CrossEntropyLoss(reduction="none")
        inputs = torch.randn(64, 64, requires_grad=True)
        targets = torch.randint(0, 64, (64,))
        self.check(ce, inputs, targets)
