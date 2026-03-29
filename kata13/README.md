# Mixture of Experts

## Task

Implement a Mixture of Experts (MoE) layer based on the DeepSeek architecture.

Implement the following in `moe.py`:

1. **MLP**: A gated MLP with SiLU activation (`w1`, `w2`, `w3`).
2. **Gate**: A routing gate that computes top-k expert selection with sigmoid scoring and weight normalization.
3. **MoE**: The full MoE module with routed experts, a shared expert, and gated routing.

The implementation will be compared against a hidden DeepSeek MoE reference in `sol/deepseek_moe.py`.

See `args.py` for the `ModelArgs` dataclass with all default hyperparameters. However, note the test config:
```
return ModelArgs(
    vocab_size=100,
    dim=32,
    inter_dim=128,
    moe_inter_dim=22,
    n_routed_experts=8,
    n_shared_experts=2,
    n_activated_experts=3,
    n_expert_groups=1,
    n_limited_groups=1,
    score_func="sigmoid",
    route_scale=1.0,
)
```