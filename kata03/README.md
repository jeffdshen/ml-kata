# Logistic SGD

## Task

Derive multinomial logistic regression training step from scratch with SGD.

It will be compared to a PyTorch implementation using nn modules.

For full points, handle both dim = 1 and dim = 2 cases (batch, input_size).

Implement:

```
# W = (out, in), bias = (out,), x = (*, in), y = (*,)
# update weight and bias
def step(weight, bias, x, y, lr):
    pass
```

For extra points, also handle the dim = 3 case (with first two dims as batch dimensions).

Another bonus, what if we take a weighted sum instead where y is an arbitrary vector?

```
# loss = (-F.log_softmax(self.linear(x)) * y).sum()
def soft_step(weight, bias, x, y, lr):
    pass
```