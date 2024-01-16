import torch
import torch.nn as nn

import torch.nn.functional as F


class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, kdim, vdim):
        super().__init__()
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(kdim, embed_dim)
        self.lin_v = nn.Linear(vdim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def get_param_mapping(self):
        return {
            "q_proj_weight": "lin_q.weight",
            "k_proj_weight": "lin_k.weight",
            "v_proj_weight": "lin_v.weight",
            "q_proj_bias": "lin_q.bias",
            "k_proj_bias": "lin_k.bias",
            "v_proj_bias": "lin_v.bias",
            "out_proj.weight": "out_proj.weight",
            "out_proj.bias": "out_proj.bias",
        }

    def to_multi_head(self, x):
        x = x.reshape(*x.size()[:-1], self.num_heads, self.embed_dim // self.num_heads)
        x = x.unsqueeze(0).transpose(0, -2).squeeze(-2)
        return x

    def from_multi_head(self, x):
        x = x.unsqueeze(-2).transpose(0, -2).squeeze(0)
        x = x.reshape(*x.size()[:-2], self.embed_dim)
        return x

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        q = self.to_multi_head(self.lin_q(query))
        k = self.to_multi_head(self.lin_k(key))
        v = self.to_multi_head(self.lin_v(value))

        x = q @ k.transpose(-1, -2)

        if key_padding_mask is not None:
            x[key_padding_mask.unsqueeze(-2).expand_as(x)] = float("-inf")
        if attn_mask is not None:
            x.masked_fill_(attn_mask, float("-inf"))
        x /= (self.embed_dim // self.num_heads) **0.5
        x = F.softmax(x, dim=-1)
        x = x @ v

        x = self.from_multi_head(x)

        x = self.out_proj(x)
        return x
