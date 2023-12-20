import unittest

import torch
import torch.nn as nn

import kata3.logistic as sol

class LogisticModel(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.lin = nn.Linear(in_size, out_size)
        self.loss = nn.CrossEntropyLoss(reduction="sum")

    def copy_params(self):
        return self.lin.weight.data.detach().clone(), self.lin.bias.data.detach().clone()

    def forward(self, x, y):
        x = self.lin(x)
        x = x.unsqueeze(-1).transpose(1, -1).squeeze(1)
        return self.loss(x, y)

class LogisticTestCase(unittest.TestCase):
    def check_and_step(self, model, x, y, lr, optimizer):
        weight, bias = model.copy_params()
        sol.step(weight, bias, x, y, lr)

        model(x, y).backward()
        optimizer.step()
        optimizer.zero_grad()
        
        expected_weight, expected_bias = model.copy_params()
        torch.testing.assert_close(weight, expected_weight)
        torch.testing.assert_close(bias, expected_bias)

    def test_dim1_small(self):
        torch.manual_seed(0)
        model = LogisticModel(4, 5)
        x = torch.randn(4)
        y = torch.randint(0, 5, tuple())
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        self.check_and_step(model, x, y, 0.1, optimizer)

    def test_dim1_multi(self):
        torch.manual_seed(1)
        model = LogisticModel(32, 32)
        x = torch.randn(32)
        y = torch.randint(0, 32, tuple())
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        for _ in range(10):
            self.check_and_step(model, x, y, 0.1, optimizer)

    def test_dim2_multi(self):
        torch.manual_seed(2)
        model = LogisticModel(32, 32)
        x = torch.randn(32, 32)
        y = torch.randint(0, 32, (32,))
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        for _ in range(10):
            self.check_and_step(model, x, y, 0.1, optimizer)

    def test_dim3_multi(self):
        torch.manual_seed(3)
        model = LogisticModel(16, 10)
        x = torch.randn(5, 8, 16)
        y = torch.randint(0, 10, (5, 8))
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        for _ in range(10):
            self.check_and_step(model, x, y, 0.1, optimizer)


class SoftLogisticModel(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.lin = nn.Linear(in_size, out_size)
        self.logprob = nn.LogSoftmax(dim=-1)

    def copy_params(self):
        return self.lin.weight.data.detach().clone(), self.lin.bias.data.detach().clone()

    def forward(self, x, y):
        x = self.lin(x)
        return (-self.logprob(x) * y).sum()

class SoftLogisticTestCase(unittest.TestCase):
    def check_and_step(self, model, x, y, lr, optimizer):
        weight, bias = model.copy_params()
        sol.soft_step(weight, bias, x, y, lr)
        model(x, y).backward()
        optimizer.step()
        optimizer.zero_grad()
        
        expected_weight, expected_bias = model.copy_params()
        torch.testing.assert_close(weight, expected_weight)
        torch.testing.assert_close(bias, expected_bias)

    def test_dim1_small(self):
        torch.manual_seed(0)
        model = SoftLogisticModel(4, 5)
        x = torch.randn(4)
        y = torch.rand(5)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        self.check_and_step(model, x, y, 0.1, optimizer)

    def test_dim1_multi(self):
        torch.manual_seed(1)
        model = SoftLogisticModel(32, 32)
        x = torch.randn(32)
        y = torch.rand(32)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        for _ in range(10):
            self.check_and_step(model, x, y, 0.1, optimizer)

    def test_dim2_multi(self):
        torch.manual_seed(2)
        model = SoftLogisticModel(32, 32)
        x = torch.randn(32, 32)
        y = torch.rand(32, 32)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        for _ in range(10):
            self.check_and_step(model, x, y, 0.1, optimizer)

    def test_dim3_multi(self):
        torch.manual_seed(3)
        model = SoftLogisticModel(16, 10)
        x = torch.randn(5, 8, 16)
        y = torch.rand(5, 8, 10)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        for _ in range(10):
            self.check_and_step(model, x, y, 0.1, optimizer)
