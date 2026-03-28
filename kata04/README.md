# RNN

## Task

Write an 2-layer Elman RNN with tanh non-linearity:

```
h[t] = tanh(x[t] W_ih^T + b_ih + h[t-1] W_hh^T + b_hh)
```

The hidden state is used as the input to the next layer.

Implement as a module called RNN:

```
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        pass
        
    def get_param_mapping(self):
        # For testing purposes.
        # Return a map from the param names in nn.RNN to the names in this model.
        # For example, you may return {"a.b.c": "d.e.f"} to indicate the 
        # other model's a.b.c param corresponds to this one's d.e.f
        # You should map the following params:
        # weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        # weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1

    def forward(self, input, h_0):
        # input = (L, N, input_size) or (L, input_size). Note that the sequence length is first.
        # h_0 = (num_layers, hidden_size)
        # returns output, h_n
        # output = (L, N, hidden_size) or (L, N, hidden_size)
        # h_n = (num_layers, hidden_size)
        pass
```