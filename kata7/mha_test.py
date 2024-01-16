import unittest
import torch
import torch.nn as nn

import kata7.mha as sol


class MhaTestCase(unittest.TestCase):
    def prepare_params(self, orig_params):
        params = {}
        for k, v in orig_params:
            if k == "in_proj_weight":
                a, b, c = v.split(v.size()[0] // 3, dim=0)
                params["q_proj_weight"] = a
                params["k_proj_weight"] = b
                params["v_proj_weight"] = c
            elif k == "in_proj_bias":
                a, b, c = v.split(v.size()[0] // 3, dim=0)
                params["q_proj_bias"] = a
                params["k_proj_bias"] = b
                params["v_proj_bias"] = c
            else:
                params[k] = v
        return params
    
    def gen_key_padding(self, size):
        key_padding_mask = torch.rand(size) + 0.01
        key_padding_mask /= key_padding_mask.mean(dim=-1, keepdim=True)
        return key_padding_mask > 1

    def gen_attn_mask(self, size, key_padding_mask=None):
        # select at least k to be false, guarantees non-nan product by pigeonhole
        if key_padding_mask is None:
            k = 1
        else:
            k = (key_padding_mask).sum(dim=-1).max().item() + 1
        attn_mask = torch.rand(size)
        lower, _ = attn_mask.kthvalue(k, dim=-1, keepdim=True)
        return attn_mask > torch.maximum(lower, torch.ones_like(lower) * 0.5)

    def check(
        self,
        mha: nn.MultiheadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        sol_mha = sol.MultiheadAttention(
            mha.embed_dim, mha.num_heads, kdim=mha.kdim, vdim=mha.vdim
        )

        param_map = sol_mha.get_param_mapping()
        mha_state_dict = self.prepare_params(mha.state_dict().items())
        state_dict = {param_map[k]: v for k, v in mha_state_dict.items()}
        sol_mha.load_state_dict(state_dict)

        outputs, _ = mha(
            query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        expected = {}
        expected["outputs"] = outputs.detach().clone()
        (outputs.sum() ** 2).backward()

        expected_grad_params = {
            k: v.grad.detach().clone() for k, v in mha.named_parameters()
        }
        expected_grad_params = self.prepare_params(expected_grad_params.items())

        expected[f"query.grad"] = query.grad.detach().clone()
        expected[f"key.grad"] = key.grad.detach().clone()
        expected[f"value.grad"] = value.grad.detach().clone()
        query.grad.zero_()
        key.grad.zero_()
        value.grad.zero_()

        outputs = sol_mha(
            query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        (outputs.sum() ** 2).backward()

        torch.testing.assert_close(outputs, expected["outputs"])
        torch.testing.assert_close(query.grad, expected["query.grad"])
        torch.testing.assert_close(key.grad, expected["key.grad"])
        torch.testing.assert_close(value.grad, expected["value.grad"])
        sol_params = dict(sol_mha.named_parameters())
        for k, v in expected_grad_params.items():
            torch.testing.assert_close(sol_params[param_map[k]].grad, v)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_cross_attn_small(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(8, 2, kdim=9, vdim=10, batch_first=True)
        q = torch.randn(2, 3, 8, requires_grad=True)
        k = torch.randn(2, 5, 9, requires_grad=True)
        v = torch.randn(2, 5, 10, requires_grad=True)
        self.check(mha, q, k, v, None, None)

    def test_self_attn_small(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(8, 2, batch_first=True)
        x = torch.randn(2, 3, 8, requires_grad=True)
        self.check(mha, x, x, x, None, None)

    def test_cross_attn_small_key_mask(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(8, 2, kdim=9, vdim=10, batch_first=True)
        q = torch.randn(2, 3, 8, requires_grad=True)
        k = torch.randn(2, 5, 9, requires_grad=True)
        v = torch.randn(2, 5, 10, requires_grad=True)
        key_padding_mask = self.gen_key_padding((2, 5))
        self.check(mha, q, k, v, key_padding_mask, None)

    def test_self_attn_small_key_mask(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(8, 2, batch_first=True)
        x = torch.randn(2, 3, 8, requires_grad=True)
        key_padding_mask = self.gen_key_padding((2, 3))
        self.check(mha, x, x, x, key_padding_mask, None)

    def test_cross_attn_small_key_mask(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(8, 2, kdim=9, vdim=10, batch_first=True)
        q = torch.randn(2, 3, 8, requires_grad=True)
        k = torch.randn(2, 5, 9, requires_grad=True)
        v = torch.randn(2, 5, 10, requires_grad=True)
        attn_mask = self.gen_attn_mask((3, 5))
        self.check(mha, q, k, v, None, attn_mask)

    def test_self_attn_small_attn_mask(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(8, 2, batch_first=True)
        x = torch.randn(2, 3, 8, requires_grad=True)
        attn_mask = self.gen_attn_mask((3, 3))
        self.check(mha, x, x, x, None, attn_mask)

    def test_cross_attn_small_masks(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(8, 2, kdim=9, vdim=10, batch_first=True)
        q = torch.randn(2, 3, 8, requires_grad=True)
        k = torch.randn(2, 5, 9, requires_grad=True)
        v = torch.randn(2, 5, 10, requires_grad=True)
        key_padding_mask = self.gen_key_padding((2, 5))
        attn_mask = self.gen_attn_mask((3, 5), key_padding_mask)
        self.check(mha, q, k, v, key_padding_mask, attn_mask)

    def test_self_attn_small_attn_mask(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(8, 2, batch_first=True)
        x = torch.randn(2, 3, 8, requires_grad=True)
        key_padding_mask = self.gen_key_padding((2, 3))
        attn_mask = self.gen_attn_mask((3, 3), key_padding_mask)
        self.check(mha, x, x, x, key_padding_mask, attn_mask)

    def test_cross_attn(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(32, 4, kdim=33, vdim=34, batch_first=True)
        q = torch.randn(2, 10, 32, requires_grad=True)
        k = torch.randn(2, 7, 33, requires_grad=True)
        v = torch.randn(2, 7, 34, requires_grad=True)
        self.check(mha, q, k, v, None, None)

    def test_self_attn(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(32, 4, batch_first=True)
        x = torch.randn(2, 10, 32, requires_grad=True)
        self.check(mha, x, x, x, None, None)

    def test_cross_attn_masks(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(32, 4, kdim=33, vdim=34, batch_first=True)
        q = torch.randn(2, 10, 32, requires_grad=True)
        k = torch.randn(2, 7, 33, requires_grad=True)
        v = torch.randn(2, 7, 34, requires_grad=True)
        key_padding_mask = self.gen_key_padding((2, 7))
        attn_mask = self.gen_attn_mask((10, 7), key_padding_mask)
        self.check(mha, q, k, v, key_padding_mask, attn_mask)

    def test_self_attn_masks(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(32, 4, batch_first=True)
        x = torch.randn(2, 10, 32, requires_grad=True)
        key_padding_mask = self.gen_key_padding((2, 10))
        attn_mask = self.gen_attn_mask((10, 10), key_padding_mask)
        self.check(mha, x, x, x, key_padding_mask, attn_mask)

    def test_cross_attn_unbatched_masks(self):
        torch.manual_seed(0)
        for _ in range(3):
            mha = nn.MultiheadAttention(32, 4, kdim=33, vdim=34, batch_first=True)
            q = torch.randn(10, 32, requires_grad=True)
            k = torch.randn(7, 33, requires_grad=True)
            v = torch.randn(7, 34, requires_grad=True)
            key_padding_mask = self.gen_key_padding((7,))
            attn_mask = self.gen_attn_mask((10, 7), key_padding_mask)
            self.check(mha, q, k, v, key_padding_mask, attn_mask)

    def test_self_attn_unbatched_masks(self):
        torch.manual_seed(0)
        for _ in range(3):
            mha = nn.MultiheadAttention(32, 4, batch_first=True)
            x = torch.randn(10, 32, requires_grad=True)
            key_padding_mask = self.gen_key_padding((10,))
            attn_mask = self.gen_attn_mask((10, 10), key_padding_mask)
            self.check(mha, x, x, x, key_padding_mask, attn_mask)


    # def test_mha(self):
    #     mha = nn.MultiheadAttention(32, 2, kdim=10, vdim=20, batch_first=True)
    #     x = torch.randn((2, 4, 32))
    #     k = torch.randn((2, 3, 10))
    #     v = torch.randn((2, 3, 20))
    #     print([(k, v.size()) for k, v in mha.named_parameters()])
    #     output, _ = mha(x, k, v, need_weights=False)
    # print(output.size())
