import torch


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, d_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        pass
