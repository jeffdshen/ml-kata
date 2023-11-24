import torch

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # shape: (*, X) @ (Y, X).T + (*, Y) -> (*, Y)
        ctx.save_for_backward(input, weight)
        return input @ weight.T + bias

    @staticmethod
    def backward(ctx, d_output):
        input, weight = ctx.saved_tensors
        # shape: (*, Y) @ (Y, X), (*, 1, X) * (*, Y, 1) -> (Y, X), (*, Y)
        d_weight = (input.unsqueeze(-2) * d_output.unsqueeze(-1))
        if d_weight.dim() > 2:
            d_weight = d_weight.sum(tuple(range(0, d_weight.dim() - 2)))
        return d_output @ weight, d_weight, d_output