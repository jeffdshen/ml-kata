import torch


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        # shape: (*, X) @ (Y, X).T + (*, Y) -> (*, Y)
        ctx.save_for_backward(input, weight)
        return input @ weight.T + bias

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, d_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input, weight = ctx.saved_tensors
        # shape: (*, Y) @ (Y, X), (*, 1, X) * (*, Y, 1) -> (*, Y, X), (*, Y)
        d_weight = input.unsqueeze(-2) * d_output.unsqueeze(-1)
        return d_output @ weight, d_weight, d_output
