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
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, self.dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.linear(x, self.weight)
        x = F.sigmoid(x)
        x, indices = torch.topk(x, self.topk)
        x /= x.sum(dim=-1, keepdim=True)
        return x, indices


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
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x.float())

        y = self.shared_experts(x).float()
        for i in range(len(self.experts)):
            token_idx, weight_idx = torch.nonzero(indices == i, as_tuple=True)
            if len(token_idx) == 0:
                continue

            expert = self.experts[i]
            y[token_idx] += expert(x[token_idx]) * weights[token_idx, weight_idx, None]

        return y.type_as(x).view(shape)

    def load_ds_state_dict(self, state_dict, *args, **kwargs):
        # TODO: Load HF state dict into our model. See below for reference.
        return self.load_state_dict(state_dict, *args, **kwargs)
