import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import kata2.mlp as sol

class MlpTestCase(unittest.TestCase):
    def test_xor(self):
        torch.manual_seed(0)
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        model = sol.Mlp(2, 2, 2)
        
        optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.99)
        for _ in range(400):
            scores = model(x)
            loss = model.loss(scores, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            self.assertAlmostEqual(F.cross_entropy(model(x), y).item(), 0.0, places=3)
