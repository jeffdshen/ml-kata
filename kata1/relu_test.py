import unittest
import torch
import torch.nn as nn

import numpy as np
import kata1.relu as sol

class ReluTestCase(unittest.TestCase):
    def check(self, relu, inputs):
        outputs = relu(inputs).sum() ** 2
        expected = {}
        expected["outputs"] = outputs.detach().clone()
        outputs.backward()

        expected["inputs.grad"] = inputs.grad.detach().clone()
        relu.zero_grad()
        inputs.grad.zero_()
        
        outputs = sol.ReluFunction.apply(inputs).sum() ** 2
        outputs.backward()
        torch.testing.assert_close(outputs, expected["outputs"])
        torch.testing.assert_close(inputs.grad, expected["inputs.grad"])
        torch.autograd.gradcheck(sol.ReluFunction.apply, (inputs,), fast_mode=True)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_relu_64(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        relu = nn.ReLU()
        inputs = torch.randn(64, requires_grad=True)
        self.check(relu, inputs)