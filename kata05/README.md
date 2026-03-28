# Scaled Dot Product Attention

## Task
Implement scaled dot product attention:

```
def scaled_dot_product_attention(query, key, value, attn_mask=None, scale=None) -> torch.Tensor:
    pass
```

Make sure it works for all dimensions, defaults, non-defaults, and with gradients. For attention, attention_mask as a bool tensor and not the generic float tensor is ok (with True meaning taking part in attention). Try to do the whole thing from memory (the torch documentation has spoilers, and the reference implementation is also slightly wrong!).