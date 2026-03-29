import torch


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.exp()
        x = x / x.sum(dim=-1, keepdim=True)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        (z,) = ctx.saved_tensors
        dz = grad_output
        # z_i = exp(x_i) / sum(exp(x_i))
        # dz_i/dx_i = z_i (1-z_i)
        # dz_i/dx_j = -z_i * z_j
        # dz/dx = I @ z - z @ z^T
        square = (z.unsqueeze(-1) @ (z.unsqueeze(-2) @ dz.unsqueeze(-1))).squeeze(-1)
        dz = dz * z - square
        return dz
