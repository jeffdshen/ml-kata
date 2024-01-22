import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import kata8.softmax as sol


class SoftmaxTestCase(unittest.TestCase):
    def check(self, inputs, values):
        outputs = (F.softmax(inputs, dim=-1) @ values).sum() ** 2
        expected = {}
        expected["outputs"] = outputs.detach().clone()
        outputs.backward()

        expected["inputs.grad"] = inputs.grad.detach().clone()
        inputs.grad.zero_()

        outputs = (sol.SoftmaxFunction.apply(inputs) @ values).sum() ** 2
        outputs.backward()
        torch.testing.assert_close(outputs, expected["outputs"])
        torch.testing.assert_close(inputs.grad, expected["inputs.grad"])
        torch.autograd.gradcheck(sol.SoftmaxFunction.apply, (inputs,), fast_mode=True)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_dim_1(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        inputs = torch.randn(8, requires_grad=True)
        values = torch.randn(8, 10)
        self.check(inputs, values)

    def test_dim_2(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(1)
        inputs = torch.randn(5, 8, requires_grad=True)
        values = torch.randn(8, 10)
        self.check(inputs, values)

    def test_dim_3(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(2)
        inputs = torch.randn(3, 5, 8, requires_grad=True)
        values = torch.randn(8, 10)
        self.check(inputs, values)
