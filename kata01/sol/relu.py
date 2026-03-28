import torch


class ReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, inputs: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        return (inputs > 0) * inputs

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, d_outputs: torch.Tensor
    ) -> torch.Tensor:
        (inputs,) = ctx.saved_tensors
        return (inputs > 0) * d_outputs
