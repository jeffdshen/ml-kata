from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

import numpy as np
import torch.nn as nn


@dataclass(frozen=True)
class Step:
    observation: list[int]
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class Action:
    action: int
    debug_info: Any


@dataclass(frozen=True)
class Episode:
    steps: list[Step]
    actions: list[Action]
    strs: list[str] | None = None
    imgs: list[np.ndarray] | None = None

    @cached_property
    def success(self):
        return self.steps[-1].terminated

    @cached_property
    def total_return(self):
        return sum([step.reward for step in self.steps])


class Policy(nn.Module, ABC):
    @abstractmethod
    def act(self, observation: list[int]) -> Action: ...


class ValueFunction(nn.Module, ABC):
    @abstractmethod
    def value(self, observation: list[int]) -> float: ...


class Env2D(Protocol):
    def step(self, action: int) -> Step: ...

    def reset(self) -> Step: ...

    @property
    def max_action(self) -> int: ...

    @property
    def input_dim(self) -> tuple[int, int]: ...

    def to_img(self) -> np.ndarray: ...

    def run_episode(self, policy: Policy, debug: bool = False) -> Episode:
        """Runs episode"""
        step = self.reset()
        steps = [step]
        actions = []
        strs = []
        imgs = []
        if debug:
            strs.append(str(self))
            imgs.append(self.to_img())

        while True:
            obs = step.observation
            action = policy.act(obs)
            step = self.step(action.action)
            steps.append(step)
            actions.append(action)
            if debug:
                strs.append(str(self))
                imgs.append(self.to_img())

            if step.terminated or step.truncated:
                if not debug:
                    strs = None
                    imgs = None
                return Episode(steps=steps, actions=actions, strs=strs, imgs=imgs)
