# Optimizers

## Task
Implement SGD with momentum, Adagrad, RMSProp, Adam, and AdamW.

Subclass the `torch.optim.Optimizer`` class. Here is an example with plain SGD:

```
class SGD(torch.optimizer.Optimizer):
    def __init__(self, params, lr=1e-1):
        defaults = dict(lr=lr)
        super().__init(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p -= lr * p.grad

                # variables can also be stored/accessed via self.state[p]:
                # if p not in self.state:
                #     self.state[p] = {"var": torch.zeros_like(p)}
                # var = self.state[p]

        return None
```

Implement the following APIs:

```
Momentum(params, lr, momentum)
Adagrad(params, lr, eps)
RMSprop(params, lr, alpha, eps)
Adam(params, lr, betas, eps)
AdamW(params, lr, betas, eps, weight_decay)
```

Some implementation notes:
1. For momentum, to follow the pytorch implementation, the momentum buffer should be initialized to the first gradient and not to all zeros.
2. The pytorch momentum formulation uses momentum and dampening (which also differs from some others). We just use momentum, i.e. `m <- a m + (1 - a) g`
3. For optimizers using eps, eps is applied after the sqrt of the variance.
4. Adam has bias correction!