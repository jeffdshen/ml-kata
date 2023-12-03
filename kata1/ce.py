import torch

import torch.nn.functional as F


class CeFunction(torch.autograd.Function):
    def forward(ctx, inputs, targets):
        x = inputs - inputs.max(dim=-1, keepdim=True)[0]
        x = torch.log(torch.exp(x).sum(dim=-1, keepdim=True)) - x
        x = torch.gather(x, -1, targets.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(inputs, targets)
        return x

    def backward(ctx, d_outputs):
        # d(-log(e^x / S))/dx = -1 + e^x / S, or dy/dx' = e^x / S
        inputs, targets = ctx.saved_tensors
        x = inputs - inputs.max(dim=-1, keepdim=True)[0]
        x = torch.exp(x)
        x = x / x.sum(dim=-1, keepdim=True)
        targets = targets.unsqueeze(-1)
        x.scatter_add_(-1, targets, -torch.ones_like(targets,dtype=x.dtype))

        return d_outputs.unsqueeze(-1) * x, None
