import torch
import numpy as np
import torch.nn.functional as F

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    scale=None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / np.sqrt(query.size(-1))
    
    # (..., L, K) @ (..., K, S)   -> (..., L, S)
    scores = (query @ key.transpose(-1, -2)) * scale

    # bias = torch.zeros(query.size(-2), key.size(-2), dtype=query.dtype, device=query.device)
    if attn_mask is not None:
        bias = torch.zeros_like(attn_mask, device=query.device, dtype=query.dtype)
        bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        scores += bias

    # (..., L, S) @ (..., S, V) ->  (..., L, V)
    return F.softmax(scores, dim=-1) @ value
