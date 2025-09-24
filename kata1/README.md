# MLP Gradients

## Task

Compute functions and gradients for:
1. A linear layer
2. A RELU activation
3. A log-softmax layer (without a mean reduction, without dim = 3+ case)

Implement them in the form of `torch.autograd.Function`.

For example:
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

Note that:
* forward, backward can accept/return multiple args that correspond to one another.
* save_for_backward, saved_tensors can also accept/return multiple args.
* The gradients from backward will be reduced via sum to match the forward shape,
which can be convenient.

The functions (`LinearFunction`, `ReluFunction`, `CeFunction`) will be compared against:
```
torch.nn.functional.linear(input, weight, bias) -> Tensor
torch.nn.functional.relu(input) -> Tensor
torch.nn.functional.log_softmax(input, dim=None) -> Tensor
```
