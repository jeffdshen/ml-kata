import torch


class Momentum(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p not in self.state:
                    self.state[p] = {}
                    self.state[p]["m"] = p.grad.clone()

                m = self.state[p]["m"]
                m *= momentum
                m += (1 - momentum) * p.grad
                p -= lr * m

        return None


class Adagrad(torch.optim.Optimizer):
    def __init__(self, params, lr, eps):
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p not in self.state:
                    self.state[p] = {}
                    self.state[p]["v"] = torch.zeros_like(p.grad)

                v = self.state[p]["v"]
                v += p.grad ** 2
                p -= lr * p.grad / (v ** 0.5 + eps)

        return None


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr, alpha, eps):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p not in self.state:
                    self.state[p] = {}
                    self.state[p]["v"] = torch.zeros_like(p.grad)

                v = self.state[p]["v"]
                v *= alpha
                v += (1 - alpha) * (p.grad ** 2)
                p -= lr * p.grad / (v ** 0.5 + eps)

        return None


class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p not in self.state:
                    self.state[p] = {}
                    self.state[p]["m"] = torch.zeros_like(p.grad)
                    self.state[p]["v"] = torch.zeros_like(p.grad)
                    self.state[p]["m_bias"] = 0.0
                    self.state[p]["v_bias"] = 0.0

                m = self.state[p]["m"]
                m *= beta1
                m += (1 - beta1) * p.grad

                self.state[p]["m_bias"] *= beta1
                self.state[p]["m_bias"] += (1 - beta1)
                m_bias = self.state[p]["m_bias"]

                v = self.state[p]["v"]
                v *= beta2
                v += (1 - beta2) * (p.grad ** 2)

                self.state[p]["v_bias"] *= beta2
                self.state[p]["v_bias"] += (1 - beta2)
                v_bias = self.state[p]["v_bias"]

                p -= lr * (m / m_bias) / ((v / v_bias) ** 0.5 + eps)

        return None




class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p not in self.state:
                    self.state[p] = {}
                    self.state[p]["m"] = torch.zeros_like(p.grad)
                    self.state[p]["v"] = torch.zeros_like(p.grad)
                    self.state[p]["m_bias"] = 0.0
                    self.state[p]["v_bias"] = 0.0

                m = self.state[p]["m"]
                m *= beta1
                m += (1 - beta1) * p.grad

                self.state[p]["m_bias"] *= beta1
                self.state[p]["m_bias"] += (1 - beta1)
                m_bias = self.state[p]["m_bias"]

                v = self.state[p]["v"]
                v *= beta2
                v += (1 - beta2) * (p.grad ** 2)

                self.state[p]["v_bias"] *= beta2
                self.state[p]["v_bias"] += (1 - beta2)
                v_bias = self.state[p]["v_bias"]

                p *= (1 - lr * wd)
                p -= lr * (m / m_bias) / ((v / v_bias) ** 0.5 + eps)

        return None


