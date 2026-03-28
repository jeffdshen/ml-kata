# Softmax

## Task

Implement softmax forward and backward (tricky!). The softmax dimension will be `dim=-1`

```
class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(a, b, c)
        return z
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b, c = ctx.saved_tensors
        return dz
```

It will be compared against:
```
torch.nn.functional.softmax(x, -1) -> Tensor
```
