import math
import random
from dataclasses import dataclass
from random import Random
from typing import Callable, Generic, TypeVar

import numpy as np

from kata12.maze_env import EMPTY, GOAL, FixedMazeEnv, MazeEnv, maze_obs_to_env
from kata12.types import Action, Env2D, Policy, Step
from kata12.vpg import evaluate_policy


@dataclass
class MctsNode:
    beta: float = 2**0.5
    value: float = 0
    visits: int = 0
    parent: "MctsNode | None" = None
    parent_action: "int | None" = None
    children: "list[MctsNode] | None" = None
    terminal: bool = False

    def update(self, value):
        self.visits += 1
        self.value = (
            self.value * ((self.visits - 1) / self.visits) + value / self.visits
        )

    @property
    def ucb(self) -> float:
        assert self.parent is not None
        if self.visits == 0:
            return 10000
        return (
            self.value + self.beta * (math.log(self.parent.visits) / self.visits) ** 0.5
        )

    def select(self) -> int | None:
        if self.children is None:
            return None
        key = lambda i: self.children[i].ucb if self.children is not None else 0
        return max(range(len(self.children)), key=key)

    @property
    def stats(self):
        return self.value, self.visits, self.terminal


EnvT = TypeVar("EnvT", bound=Env2D)


@dataclass
class MctsPolicy(Policy, Generic[EnvT]):
    obs_to_env: Callable[[list[int]], EnvT]
    heuristic: Callable[[EnvT], int]
    total_steps: int = 1000
    beta: float = 2**0.5
    expand_visits: int = 1
    max_rollout: int = 10
    gamma: float = 0.99

    def act(self, observation: list[int]) -> Action:
        root = MctsNode(beta=self.beta)
        env = self.obs_to_env(observation)
        for i in range(self.total_steps):
            self.step(root, env)

        assert root.children is not None
        action = max(
            list(range(len(root.children))),
            key=lambda i: root.children[i].visits if root.children is not None else 0,
        )
        logits = [
            0.0 if i == action else float("-inf") for i in range(len(root.children))
        ]
        return Action(logits)

    def print_tree(self, node: MctsNode, max_level: int, level: int = 0):
        if level > max_level:
            return
        print("  " * level + str((node.parent_action, node.value, node.visits)))
        if node.children is None:
            return
        for child in node.children:
            self.print_tree(child, max_level, level + 1)

    def step(self, root: MctsNode, env: EnvT):
        env.reset()

        path = self.select(root)
        self.expand(path, env)
        steps: list[Step] = []
        for node in path[1:]:
            assert node.parent_action is not None
            step = env.step(node.parent_action)
            steps.append(step)

        if steps and (steps[-1].terminated or steps[-1].truncated):
            value = 0.0
            path[-1].terminal = True
        else:
            value = self.rollout(env)

        returns = []
        for step in reversed(steps):
            value *= self.gamma
            value += step.reward
            returns.append(value)
        returns.append(value * self.gamma)
        returns.reverse()

        for node, r in zip(path, returns, strict=True):
            node.update(r)

    def rollout(self, env: EnvT) -> float:
        value = 0.0
        discount = 1.0
        for _ in range(self.max_rollout):
            action = self.heuristic(env)
            step = env.step(action)
            value += discount * step.reward
            discount *= self.gamma
            if step.terminated or step.truncated:
                # does assume truncated value is 0.0
                break
        return value

    def expand(self, path: list[MctsNode], env: EnvT):
        node = path[-1]
        if len(path) > 1 and node.visits < self.expand_visits:
            return
        if node.terminal:
            return

        node.children = []
        for i in range(env.max_action):
            node.children.append(MctsNode(beta=self.beta, parent=node, parent_action=i))
        action = node.select()
        if action is not None:
            path.append(node.children[action])

    def select(self, root: MctsNode) -> list[MctsNode]:
        node = root
        path = []
        while node != None:
            path.append(node)
            action = node.select()
            if action is None:
                node = None
            else:
                assert node.children is not None
                node = node.children[action]

        return path


CHOICES = [(0, (0, 1)), (1, (1, 0))]


def simple_heuristic(rng: Random, env: FixedMazeEnv) -> int:
    state = env.state
    assert state is not None
    valid_choices = []
    for choice, (dx, dy) in CHOICES:
        nx, ny = state.x + dx, state.y + dy
        if (
            0 <= nx < state.m
            and 0 <= ny < state.n
            and state.grid[nx, ny] in [EMPTY, GOAL]
        ):
            valid_choices.append(choice)

    if valid_choices:
        return rng.choice(valid_choices)

    return rng.choice([0, 1])


def main():
    random.seed(0)

    n = 10
    total_steps = 1000
    max_rollout = 30
    beta = 2**0.5 / 10
    val_env = MazeEnv(n=n, fill_prob=0.5, rng=Random(42))
    rng = Random(1337)
    heuristic = lambda x: simple_heuristic(rng, x)
    policy = MctsPolicy[FixedMazeEnv](
        obs_to_env=lambda obs: maze_obs_to_env(obs, n, val_env.max_steps),
        heuristic=heuristic,
        total_steps=total_steps,
        beta=beta,
        max_rollout=max_rollout,
    )

    metrics = evaluate_policy(
        env=val_env,
        policy=policy,
        num_episodes=100,
        num_save_gifs=10,
        # num_episodes=2,
        # num_save_gifs=2,
        output_dir="maze_output/mcts",
    )
    print("Evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
