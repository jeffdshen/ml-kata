import unittest
import torch
import torch.nn as nn
import numpy as np
import kata1.linear as sol


class LinearTestCase(unittest.TestCase):
    def check(self, linear, inputs):
        outputs = linear(inputs).sum() ** 2
        expected = {}
        expected["outputs"] = outputs.detach().clone()
        outputs.backward()

        expected["weight.grad"] = linear.weight.grad.detach().clone()
        expected["bias.grad"] = linear.bias.grad.detach().clone()
        expected["inputs.grad"] = inputs.grad.detach().clone()
        linear.zero_grad()
        inputs.grad.zero_()

        outputs = sol.LinearFunction.apply(inputs, linear.weight, linear.bias).sum() ** 2
        outputs.backward()
        torch.testing.assert_close(outputs, expected["outputs"])
        torch.testing.assert_close(linear.weight.grad, expected["weight.grad"])
        torch.testing.assert_close(linear.bias.grad, expected["bias.grad"])
        torch.testing.assert_close(inputs.grad, expected["inputs.grad"])

        torch.autograd.gradcheck(
            sol.LinearFunction.apply,
            (inputs, linear.weight, linear.bias),
            fast_mode=True,
        )

    def test_dims_1(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        linear = nn.Linear(64, 64)
        inputs = torch.randn(64, requires_grad=True)
        self.check(linear, inputs)

    def test_dims_2(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        linear = nn.Linear(64, 64)
        inputs = torch.randn(64, 64, requires_grad=True)
        self.check(linear, inputs)

    def test_dims_3(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        linear = nn.Linear(64, 64)
        inputs = torch.randn(20, 64, 64)
        inputs.requires_grad_(True)
        self.check(linear, inputs)
