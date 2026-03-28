import torch


def step(
    weight: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
) -> None:
    # More generally,
    # grad_z = yhat * (sum(y)) - y
    # when sum(y) = 1, we get the normal case.
    z = x @ weight.T + bias
    exp_z = torch.exp(z - z.max(dim=-1, keepdim=True)[0])
    yhat = exp_z / exp_z.sum(dim=-1, keepdim=True)

    grad_z = yhat * y.sum(dim=-1, keepdim=True) - y
    grad_zt = grad_z.unsqueeze(0).transpose(0, -1).squeeze(-1)
    grad_w = torch.tensordot(grad_zt, x, dims=x.dim() - 1)
    weight -= lr * grad_w
    bias -= lr * grad_z.unsqueeze(0).flatten(0, -2).sum(dim=0)
