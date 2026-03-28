# RNN

## Task

Write an 2-layer Elman RNN with tanh non-linearity:

```
h[t] = tanh(x[t] W_ih^T + b_ih + h[t-1] W_hh^T + b_hh)
```

The hidden state is used as the input to the next layer. Implement as a module called RNN.