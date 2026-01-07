import torch
import torch.nn as nn
import torch.nn.functional as F

from kata13.args import ModelArgs


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.bfloat16)
        self.w2 = nn.Linear(inter_dim, dim, bias=False, dtype=torch.bfloat16)
        self.w3 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor):
        pass


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, self.dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.gate = Gate(args)
        self.experts = nn.ModuleList(
            [MLP(args.dim, args.moe_inter_dim) for _ in range(args.n_routed_experts)]
        )
        self.shared_experts = MLP(args.dim, args.moe_inter_dim * args.n_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def load_ds_state_dict(self, state_dict, *args, **kwargs):
        # TODO: Load HF state dict into our model. See below for reference.
        return self.load_state_dict(state_dict, *args, **kwargs)


raise NotImplementedError("kata13/llama.py not implemented")
