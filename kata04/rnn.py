import torch
import torch.nn as nn


class RNN(nn.Module):
    """A 2-layer Elman RNN with tanh non-linearity.

    h[t] = tanh(x[t] W_ih^T + b_ih + h[t-1] W_hh^T + b_hh)

    The hidden state is used as the input to the next layer.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        pass

    def get_param_mapping(self) -> dict[str, str]:
        """Return a map from nn.RNN param names to this model's param names.

        For example, {"a.b.c": "d.e.f"} indicates the other model's
        a.b.c param corresponds to this one's d.e.f.

        You should map the following params:
        weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1
        """
        pass

    def forward(
        self, input: torch.Tensor, h_0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input: (L, N, input_size) or (L, input_size). Sequence length is first.
            h_0: (num_layers, hidden_size) or (num_layers, N, hidden_size).

        Returns:
            output: (L, N, hidden_size) or (L, hidden_size).
            h_n: (num_layers, hidden_size) or (num_layers, N, hidden_size).
        """
        pass
