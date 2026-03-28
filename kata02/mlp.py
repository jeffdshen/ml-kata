import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        pass

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass
