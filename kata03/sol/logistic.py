import torch


def step(
    weight: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
) -> None:
    # L = sum(y * -log(yhat)), yhat = softmax(z), z = W @ x + b
    # where softmax = e^z_i / sum(e^z_j)
    # d(log sum(e^z_j))/dz = e^z_i/sum(e^z_j) = softmax(z)
    # grad_z = yhat - y.
    # grad_W = grad_z @ grad_z_over_W  = (yhat - y) @ x^T, the outer product
    # grad_b = grad_z = yhat - y

    # Since batch size is first...
    z = x @ weight.T + bias
    exp_z = torch.exp(z - z.max(dim=-1, keepdim=True)[0])
    yhat = exp_z / exp_z.sum(dim=-1, keepdim=True)

    grad_z = yhat
    y = y.unsqueeze(-1)
    grad_z.scatter_add_(-1, y, -torch.ones_like(y, dtype=grad_z.dtype))
    grad_zt = grad_z.unsqueeze(0).transpose(0, -1).squeeze(-1)
    grad_w = torch.tensordot(grad_zt, x, dims=x.dim() - 1)
    weight -= lr * grad_w
    bias -= lr * grad_z.unsqueeze(0).flatten(0, -2).sum(dim=0)
