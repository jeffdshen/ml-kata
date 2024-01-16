# Multi-Head Attention

## Task

Implement multi-head attention:

```
class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, kdim, vdim):
        pass

    def get_param_mapping(self):
        # for testing purposes, provide a mapping the torch implementation's variables to this one
        return {
            "q_proj_weight": "",
            "k_proj_weight": "",
            "v_proj_weight": "",
            "in_proj_bias": "",
            "out_proj.weight": "",
            "out_proj.bias": "",
        }

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # Returns: the output of multihead attention, a single tensor
        pass
```

Make sure it works for all dimensions (batched and unbatched), defaults, non-defaults, and with gradients. Several notes (with several constraints added for simplicity):
1. `embed_dim` will be an integer multiple of `num_heads`
2. In some tests, `kdim`, `vdim`, and `embed_dim` may all be different.
3. No dropout.
4. query, key, value will be batch first.
5. key and value will be of same sequence length, but be different from query's.
6. `key_padding_mask` and `attn_mask` will be bool tensors (with True meaning **not** taking part in attention)
7. `attn_mask` will not be more than 2 dimensions, i.e. `(query_length, key_length)`, but not `(batch_size * num_heads, query_length, key_length)`.