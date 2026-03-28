import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        loss = F.cross_entropy(x, targets)
        return loss, x
