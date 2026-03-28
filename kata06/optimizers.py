from collections.abc import Callable, Iterable

import torch


class Momentum(torch.optim.Optimizer):
    def __init__(
        self, params: Iterable[torch.Tensor], lr: float, momentum: float
    ) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        pass


class Adagrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], lr: float, eps: float) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        pass


class RMSprop(torch.optim.Optimizer):
    def __init__(
        self, params: Iterable[torch.Tensor], lr: float, alpha: float, eps: float
    ) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        pass


class Adam(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        pass


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        pass
