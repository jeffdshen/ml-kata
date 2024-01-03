import unittest

import torch
import torch.nn as nn

import kata6.optimizers as sol


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(5, 7)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        x = inputs
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return self.loss(x, targets)


def step_and_check(model, optimizer, ref_model, ref_optimizer, inputs, targets):
    model.load_state_dict(ref_model.state_dict())
    loss = ref_model(inputs, targets)
    loss.backward()
    ref_optimizer.step()
    ref_optimizer.zero_grad()

    loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    param_dict = {name: params for name, params in model.named_parameters()}
    for name, params in ref_model.named_parameters():
        torch.testing.assert_close(param_dict.get(name), params)


class MomentumTestCase(unittest.TestCase):
    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_steps_10(self):
        torch.manual_seed(0)
        model = DummyModel()
        ref_model = DummyModel()
        lr = 0.1
        momentum = 0.9

        optimizer = sol.Momentum(model.parameters(), lr, momentum)
        ref_optimizer = torch.optim.SGD(
            ref_model.parameters(), lr, momentum=momentum, dampening=momentum
        )
        inputs = torch.randn(16, 10)
        targets = torch.randint(0, 7, (16,))
        for _ in range(10):
            step_and_check(model, optimizer, ref_model, ref_optimizer, inputs, targets)


class AdagradTestCase(unittest.TestCase):
    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_steps_10(self):
        torch.manual_seed(0)
        model = DummyModel()
        ref_model = DummyModel()
        lr = 0.01
        eps = 1e-6

        optimizer = sol.Adagrad(model.parameters(), lr, eps)
        ref_optimizer = torch.optim.Adagrad(ref_model.parameters(), lr, eps=eps)
        inputs = torch.randn(16, 10)
        targets = torch.randint(0, 7, (16,))
        for _ in range(10):
            step_and_check(model, optimizer, ref_model, ref_optimizer, inputs, targets)


class RMSpropTestCase(unittest.TestCase):
    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_steps_10(self):
        torch.manual_seed(0)
        model = DummyModel()
        ref_model = DummyModel()
        lr = 0.01
        alpha = 0.99
        eps = 1e-6

        optimizer = sol.RMSprop(model.parameters(), lr, alpha, eps)
        ref_optimizer = torch.optim.RMSprop(
            ref_model.parameters(), lr=lr, alpha=alpha, eps=eps
        )
        inputs = torch.randn(16, 10)
        targets = torch.randint(0, 7, (16,))
        for _ in range(10):
            step_and_check(model, optimizer, ref_model, ref_optimizer, inputs, targets)


class AdamTestCase(unittest.TestCase):
    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_steps_10(self):
        torch.manual_seed(0)
        model = DummyModel()
        ref_model = DummyModel()
        lr = 0.01
        betas = [0.9, 0.99]
        eps = 1e-8

        optimizer = sol.Adam(model.parameters(), lr, betas, eps)
        ref_optimizer = torch.optim.Adam(
            ref_model.parameters(), lr=lr, betas=betas, eps=eps
        )
        inputs = torch.randn(16, 10)
        targets = torch.randint(0, 7, (16,))
        for _ in range(10):
            step_and_check(model, optimizer, ref_model, ref_optimizer, inputs, targets)


class AdamWTestCase(unittest.TestCase):
    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_steps_10(self):
        torch.manual_seed(0)
        model = DummyModel()
        ref_model = DummyModel()
        lr = 0.01
        betas = [0.9, 0.99]
        eps = 1e-8
        weight_decay = 0.01

        optimizer = sol.AdamW(model.parameters(), lr, betas, eps, weight_decay)
        ref_optimizer = torch.optim.AdamW(
            ref_model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        inputs = torch.randn(16, 10)
        targets = torch.randint(0, 7, (16,))
        for _ in range(10):
            step_and_check(model, optimizer, ref_model, ref_optimizer, inputs, targets)
