import torch


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        pass

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        pass
