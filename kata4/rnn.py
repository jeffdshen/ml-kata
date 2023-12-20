import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lin_ih_l0 = nn.Linear(input_size, hidden_size)
        self.lin_ih_l1 = nn.Linear(hidden_size, hidden_size)
        self.lin_hh_l0 = nn.Linear(hidden_size, hidden_size)
        self.lin_hh_l1 = nn.Linear(hidden_size, hidden_size)

    def get_param_mapping(self):
        return {
            "weight_ih_l0": "lin_ih_l0.weight",
            "bias_ih_l0": "lin_ih_l0.bias",
            "weight_hh_l0": "lin_hh_l0.weight",
            "bias_hh_l0": "lin_hh_l0.bias",
            "weight_ih_l1": "lin_ih_l1.weight",
            "bias_ih_l1": "lin_ih_l1.bias",
            "weight_hh_l1": "lin_hh_l1.weight",
            "bias_hh_l1": "lin_hh_l1.bias"
        }

    def forward(self, input, h_0):
        # input = (L, N, input_size) or (L, input_size). Note that the sequence length is first.
        # h_0 = (num_layers, hidden_size)
        # returns output, h_n
        # output = (L, N, hidden_size) or (L, N, hidden_size)
        # h_n = (num_layers, hidden_size)
        h_n_0, h_n_1 = torch.unbind(h_0)
        outputs = []
        for x in torch.unbind(input):
            h_n_0 = F.tanh(self.lin_ih_l0(x) + self.lin_hh_l0(h_n_0))
            h_n_1 = F.tanh(self.lin_ih_l1(h_n_0) + self.lin_hh_l1(h_n_1))
            outputs.append(h_n_1)

        return torch.stack(outputs), torch.stack([h_n_0, h_n_1])
