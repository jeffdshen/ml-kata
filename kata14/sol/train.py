import numpy as np
import torch


def train(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    valid_inputs: np.ndarray,
    valid_targets: np.ndarray,
    output_size: int,
    epochs: int,
    verbose: bool = False,
) -> np.ndarray:
    train_inputs = torch.from_numpy(train_inputs)
    train_targets = torch.from_numpy(train_targets)
    valid_inputs = torch.from_numpy(valid_inputs)
    valid_targets = torch.from_numpy(valid_targets)
    input_size = train_inputs.shape[-1]
    hidden_size = 256
    batch_size = 16
    w1 = torch.randn(hidden_size, input_size) / (input_size**0.5)
    b1 = torch.zeros(hidden_size)
    w2 = torch.randn(output_size, hidden_size) / (hidden_size**0.5)
    b2 = torch.zeros(output_size)

    def forward(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(torch.float32)
        # x1: (b, input_size)
        x1 = x
        x = x @ w1.T + b1
        x = (x >= 0) * x

        # x2: (b, hidden_size)
        x2 = x
        x = x @ w2.T + b2
        x = x - x.max(dim=-1, keepdim=True)[0]
        x = x - x.exp().sum(dim=-1, keepdim=True).log()
        z = x
        x = x.exp()

        return x1, x2, z, x

    lr = 0.01
    for _ in range(epochs):
        losses = []
        shuffle = torch.randperm(train_inputs.size(0))
        inputs = train_inputs[shuffle]
        targets = train_targets[shuffle]
        for b in range(0, len(train_inputs), batch_size):
            x = inputs[b : b + batch_size]
            target = targets[b : b + batch_size]
            y = torch.eye(output_size)[target]

            x1, x2, z, x = forward(x)

            losses.append((-z.gather(-1, target.unsqueeze(-1))).mean().item())
            # dz: (b, output_size)
            dz = (x - y) / x.size(0)
            db2 = dz.sum(dim=0)
            dw2 = dz.T @ x2

            # dx2: (b, hidden_size)
            dx2 = dz @ w2
            dx2 = dx2 * (x2 > 0)
            db1 = dx2.sum(dim=0)
            dw1 = dx2.T @ x1

            w1 -= lr * dw1
            w2 -= lr * dw2
            b1 -= lr * db1
            b2 -= lr * db2
        if verbose:
            x1, x2, z, x = forward(valid_inputs)
            valid_loss = (-z.gather(-1, valid_targets.unsqueeze(-1))).mean().item()
            print(f"train_loss: {np.mean(losses)}, valid_loss: {valid_loss}")

    x1, x2, z, x = forward(valid_inputs)
    return x.argmax(dim=-1).numpy()
