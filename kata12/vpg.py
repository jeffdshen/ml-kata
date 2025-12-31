import dataclasses
import json
import random
import time
from pathlib import Path
from random import Random

import torch
import torch.optim as optim
from PIL import Image

from kata12.maze_env import *
from kata12.models import *
from kata12.types import *


def vpg_train(
    env: Env2D,
    policy: Policy,
    optimizer: optim.Optimizer,
    vf: ValueFunction,
    vf_optimizer: optim.Optimizer,
    epochs: int,
    episodes_per_epoch: int,
    vf_steps_per_epoch: int,
    gamma: float,
    gae_lambda: float,
) -> Policy:
    total_return_ema = 0.0
    returns_ema = 0.0
    steps_ema = 0.0
    success_ema = 0.0
    loss_v_ema = 0.0
    norm = 0.0

    start_time = time.time()

    for epoch in range(epochs):
        obs_t = []
        returns_t = []
        actions_t = []
        advantages_t = []

        for ep in range(episodes_per_epoch):
            total_ep = epoch * episodes_per_epoch + ep
            episode = env.run_episode(policy)
            returns = []
            advantages = []
            values = [
                vf.value(step.observation) if not step.terminated else 0.0
                for step in episode.steps
            ]
            gae_rewards = [
                step.reward + gamma * value - prev_value
                for prev_value, value, step in zip(
                    values[1:], values[:-1], episode.steps[1:]
                )
            ]

            total_return = sum(step.reward for step in episode.steps)
            G = values[-1]
            for step in reversed(episode.steps):
                G = step.reward + gamma * G
                returns.append(G)
            returns.pop()
            returns.reverse()

            G = values[-1]
            for gae_reward in reversed(gae_rewards):
                G = gae_reward + gae_lambda * gamma * G
                advantages.append(G)
            advantages.reverse()

            returns_ema = 0.99 * returns_ema + 0.01 * returns[0]
            total_return_ema = 0.99 * total_return_ema + 0.01 * total_return
            steps_ema = 0.99 * steps_ema + 0.01 * len(episode.steps)
            success_ema = 0.99 * success_ema + 0.01 * episode.success

            if total_ep % 100 == 0:
                weight_norm = torch.norm(
                    torch.stack([p.norm(2) for p in policy.parameters()]), 2
                )
                elapsed_time = time.time() - start_time
                print(
                    total_ep,
                    f"time={elapsed_time:.4f}",
                    f"r={total_return_ema:.4f}",
                    f"r_gamma={returns_ema:.4f}",
                    f"steps={steps_ema:.4f}",
                    f"success={success_ema:.4f}",
                    f"{norm=:.4f}",
                    f"{weight_norm=:.4f}",
                )

            for step, action, r, adv in zip(
                episode.steps[:-1], episode.actions, returns, advantages, strict=True
            ):
                obs_t.append(torch.tensor(step.observation, dtype=torch.float32))
                returns_t.append(torch.tensor(r, dtype=torch.float32))
                advantages_t.append(torch.tensor(adv, dtype=torch.float))
                actions_t.append(torch.tensor(action.action, dtype=torch.long))

        obs_t, actions_t, returns_t, advantages_t = (
            torch.stack(obs_t),
            torch.stack(actions_t),
            torch.stack(returns_t),
            torch.stack(advantages_t),
        )
        returns_t = (returns_t - returns_t.mean()) / (
            returns_t.std(unbiased=False) + 1e-8
        )
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std(unbiased=False) + 1e-8
        )
        logits_t = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits_t)
        logp: torch.Tensor = dist.log_prob(actions_t)
        # loss = (-logp * returns_t).sum() / episodes_per_epoch
        loss = (-logp * returns_t).mean()
        # loss = (-logp * advantages_t).mean()

        loss.backward()
        try:
            norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=5.0, error_if_nonfinite=True
            )
        except:
            print(loss)
            print(returns_t)
            print(logp)
            print(actions_t)
            print(obs_t)
            raise

        optimizer.step()
        optimizer.zero_grad()

        for _ in range(vf_steps_per_epoch):
            vf_optimizer.zero_grad()
            value: torch.Tensor = vf(obs_t)
            loss_v = ((value - returns_t) ** 2).mean()
            loss_v.backward()
            vf_optimizer.step()

    return policy


def evaluate_policy(
    env: Env2D,
    policy: Policy,
    num_episodes: int = 100,
    num_save_gifs: int = 10,
    output_dir: str | None = None,
) -> dict[str, float]:
    episodes: list[Episode] = []

    for _ in range(num_episodes):
        episode = env.run_episode(policy, debug=True)
        episodes.append(episode)

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
                    f.write(
                        f"Step {i}: "
                        f"{action.action}, "
                        f"{np.round(action.debug_info['probs'], 4)}, "
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


def main():
    random.seed(0)
    torch.manual_seed(0)

    n = 10
    total_episodes = 50000
    episodes_per_epoch = 100
    epochs = total_episodes // episodes_per_epoch

    env = MazeEnv(
        n=n, fill_prob=0.5, rng=Random(42), max_repeats=1, randomize_agent=False
    )
    val_env = MazeEnv(n=n, fill_prob=0.5, rng=Random(42))
    # policy = MlpPolicy(
    #     input_dim=env.input_dim[0] * env.input_dim[1], action_dim=env.max_action
    # )
    # vf = MlpValue(input_dim=env.input_dim[0] * env.input_dim[1])
    policy = ConvPolicy(
        input_dim=env.input_dim, action_dim=env.max_action, residual=True, ffn_dim=32
    )
    # vf = ConvValue(input_dim=env.input_dim)
    vf = DummyValue()

    optimizer = optim.AdamW(policy.parameters(), lr=1e-3)
    vf_optimizer = optim.AdamW(policy.parameters(), lr=1e-3)
    # optimizer = optim.SGD(policy.parameters(), lr=1e-3, momentum=0.9)
    # vf_optimizer = optim.SGD(vf.parameters(), lr=1e-3, momentum=0.9)
    try:
        policy = vpg_train(
            env=env,
            policy=policy,
            optimizer=optimizer,
            vf=vf,
            vf_optimizer=vf_optimizer,
            epochs=epochs,
            episodes_per_epoch=episodes_per_epoch,
            vf_steps_per_epoch=4,
            gamma=0.99,
            gae_lambda=0.95,
        )
    except KeyboardInterrupt:
        print("stopped")
    except:
        with open(Path("maze_output") / "policy.pt", "wb") as f:
            torch.save(policy.state_dict(), f)
            print(env)
            assert env.state is not None
            print(env.state.grid)
        raise

    metrics = evaluate_policy(
        env=val_env,
        policy=policy,
        num_episodes=100,
        num_save_gifs=10,
        output_dir="maze_output",
    )
    print("Evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
