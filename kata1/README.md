# MLP Gradients

## Task

Compute functions and gradients for:
1. A linear layer
2. A RELU activation
3. A log-softmax layer (without a mean reduction)

Implement them in the form of `torch.autograd.Function`.

For example (note these can accept multiple args):
```
class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

     @staticmethod
     def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result
```

They will be compared against:
```
torch.nn.functional.linear(input, weight, bias) -> Tensor
torch.nn.functional.relu(input) -> Tensor
torch.nn.functional.log_softmax(input, dim=None) -> Tensor
```
