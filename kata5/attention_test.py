import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import kata5.attention as sol

class AttentionTestCase(unittest.TestCase):
    def check(self, query, key, value, attn_mask=None, scale=None):
        expected = {}
        outputs = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, scale=scale)
        expected["outputs"] = outputs.detach().clone()
        loss = outputs.sum() ** 2
        loss.backward()
        expected["query.grad"] = query.grad.detach().clone()
        expected["key.grad"] = key.grad.detach().clone()
        expected["value.grad"] = value.grad.detach().clone()

        query.grad.zero_()
        key.grad.zero_()
        value.grad.zero_()
        
        outputs = sol.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, scale=scale)
        loss = outputs.sum() ** 2
        loss.backward()

        torch.testing.assert_close(outputs, expected["outputs"])
        torch.testing.assert_close(query.grad, expected["query.grad"])
        torch.testing.assert_close(key.grad, expected["key.grad"])
        torch.testing.assert_close(value.grad, expected["value.grad"])


    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_dim_2(self):
        torch.manual_seed(0)
        query = torch.randn(10, 64, requires_grad=True)
        key = torch.randn(15, 64, requires_grad=True)
        value = torch.randn(15, 128, requires_grad=True)
        self.check(query, key, value)

    def test_dim_3(self):
        torch.manual_seed(0)
        query = torch.randn(3, 10, 64, requires_grad=True)
        key = torch.randn(3, 15, 64, requires_grad=True)
        value = torch.randn(3, 15, 128, requires_grad=True)
        self.check(query, key, value)

    def test_mask_dim_2(self):
        torch.manual_seed(0)
        attn_mask = torch.rand(10, 15) > 0.5
        query = torch.randn(3, 10, 64, requires_grad=True)
        key = torch.randn(3, 15, 64, requires_grad=True)
        value = torch.randn(3, 15, 128, requires_grad=True)
        self.check(query, key, value, attn_mask=attn_mask)

    def test_mask_dim_3(self):
        torch.manual_seed(0)
        attn_mask = torch.rand(3, 10, 15) > 0.5
        query = torch.randn(3, 10, 64, requires_grad=True)
        key = torch.randn(3, 15, 64, requires_grad=True)
        value = torch.randn(3, 15, 128, requires_grad=True)
        self.check(query, key, value, attn_mask=attn_mask)

    def test_scale(self):
        torch.manual_seed(0)
        query = torch.randn(3, 10, 64, requires_grad=True)
        key = torch.randn(3, 15, 64, requires_grad=True)
        value = torch.randn(3, 15, 128, requires_grad=True)
        scale = 3.14
        self.check(query, key, value, scale=scale)

    def test_all_together_now(self):
        torch.manual_seed(0)
        attn_mask = torch.rand(3, 15, 10) > 0.5
        query = torch.randn(3, 15, 32, requires_grad=True)
        key = torch.randn(3, 10, 32, requires_grad=True)
        value = torch.randn(3, 10, 64, requires_grad=True)
        scale = 3.14
        self.check(query, key, value, attn_mask=attn_mask, scale=scale)