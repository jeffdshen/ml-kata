import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, kdim: int, vdim: int) -> None:
        pass

    def get_param_mapping(self) -> dict[str, str]:
        """
        for testing purposes, provide a mapping the torch implementation's variables to this one
        """
        return {
            "q_proj_weight": "",
            "k_proj_weight": "",
            "v_proj_weight": "",
            "in_proj_bias": "",
            "out_proj.weight": "",
            "out_proj.bias": "",
        }

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns: the output of multihead attention, a single tensor
        """
