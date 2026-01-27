"""
This file is derived from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
Copyright (c) 2018 OpenAI (http://openai.com)
Copyright (c) 2026 jeffdshen

Licensed under the MIT License.
See LICENSE or https://opensource.org/licenses/MIT
"""

from typing import Protocol

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from kata12.models import ConvPolicy, ConvValue


class ActorCritic(Protocol):
    @property
    def pi(self) -> nn.Module: ...

    @property
    def v(self) -> nn.Module: ...

    def step(self, obs: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(
            self.v_net(obs), -1
        )  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_sizes=(64, 64), activation=nn.Tanh
    ):
        super().__init__()

        self.pi = MLPCategoricalActor(obs_dim, action_dim, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class ConvCategoricalActor(Actor):
    def __init__(self, input_dim, action_dim, **kwargs):
        super().__init__()
        self.policy = ConvPolicy(input_dim, action_dim, **kwargs)

    def _distribution(self, obs):
        return Categorical(logits=self.policy(obs))

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class ConvCritic(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        self.vf = ConvValue(input_dim, **kwargs)

    def forward(self, obs):
        return self.vf(obs)


class ConvActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, **kwargs):
        super().__init__()
        self.pi = ConvCategoricalActor(input_dim, action_dim, **kwargs)
        self.v = ConvCritic(input_dim, **kwargs)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
