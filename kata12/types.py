import dataclasses
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch.nn as nn
from PIL import Image


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


def evaluate_policy(
    env: Env2D,
    policy: Policy,
    num_episodes: int = 100,
    num_save_gifs: int = 10,
    output_dir: str | None = None,
    action_debug_print_fn: Callable | None = None,
) -> dict[str, float]:
    episodes: list[Episode] = []

    print("evaluating...")
    for i in range(num_episodes):
        episode = env.run_episode(policy, debug=True)
        episodes.append(episode)
        print(
            f"finished episode {i}: {episode.success=}, {episode.total_return=}, {len(episode.steps)}"
        )

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for i, ep in enumerate(episodes[:num_save_gifs]):
            assert ep.imgs is not None
            imgs = [
                np.repeat(np.repeat(img, 20, axis=0), 20, axis=1) for img in ep.imgs
            ]
            frames = [Image.fromarray(img) for img in imgs]
            gif_path = output_path / f"episode_{i:03d}.gif"
            frames[0].save(
                gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0
            )
        episode_jsons = [
            dataclasses.asdict(dataclasses.replace(ep, imgs=None)) for ep in episodes
        ]
        with open(output_path / "episodes.json", "w") as f:
            json.dump(episode_jsons, f)

        with open(output_path / "episodes.log", "w") as f:
            for i, ep in enumerate(episodes):
                f.write(f"Episode: {i:03d}\n")
                assert ep.strs is not None
                f.write(ep.strs[0])
                for i, (step, action, s) in enumerate(
                    zip(ep.steps[1:], ep.actions, ep.strs[1:])
                ):
                    debug_info = (
                        action_debug_print_fn(action.debug_info)
                        if action_debug_print_fn is not None
                        else None
                    )

                    f.write(
                        f"Step {i}: "
                        f"{action.action}, "
                        f"{debug_info}, "
                        f"{step.reward}\n"
                    )
                    f.write(s)
                f.write(f"{'=' * 30}\n\n")
        print(f"Saved to: {output_dir}")

    return {
        "success_rate": float(np.mean([e.success for e in episodes])),
        "avg_return": float(np.mean([e.total_return for e in episodes])),
        "avg_steps": float(np.mean([len(e.steps) for e in episodes])),
    }
