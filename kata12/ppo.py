import random
import time
from pathlib import Path
from random import Random

import torch
import torch.optim as optim

from kata12.maze_env import *
from kata12.models import *
from kata12.types import *


def to_returns(rewards: list[float], value: float, discount_factor: float):
    G = value
    returns = []
    for reward in reversed(rewards):
        G = reward + discount_factor * G
        returns.append(G)
    returns.reverse()
    return returns


def ppo_train(
    env: Env2D,
    policy: Policy,
    optimizer: optim.Optimizer,
    vf: ValueFunction,
    vf_optimizer: optim.Optimizer,
    epochs: int,
    episodes_per_epoch: int,
    pi_steps_per_epoch: int,
    vf_steps_per_epoch: int,
    gamma: float,
    gae_lambda: float,
    clip_ratio: float,
):
    stats = EmaStats()
    start_time = time.time()
    norm = 0.0

    for epoch in range(epochs):
        obs_t = []
        returns_t = []
        actions_t = []
        old_logits_t = []
        advantages_t = []

        for ep in range(episodes_per_epoch):
            total_ep = epoch * episodes_per_epoch + ep
            episode = env.run_episode(policy)
            returns = []
            advantages = []
            # TODO: should truncated be omitted here?
            values = [
                vf.value(step.observation) if not (step.terminated or step.truncated) else 0.0
                for step in episode.steps
            ]
            rewards = [step.reward for step in episode.steps[1:]]
            gae_rewards = [
                reward + gamma * value - prev_value
                for prev_value, value, reward in zip(values[:-1], values[1:], rewards)
            ]
            returns = to_returns(rewards, values[-1], gamma)
            advantages = to_returns(gae_rewards, 0, gae_lambda * gamma)

            total_return = sum(step.reward for step in episode.steps)
            stats.log_ema(
                {
                    "R": total_return,
                    "r": returns[0],
                    "steps": len(episode.steps),
                    "success": episode.success,
                },
                ratio=0.99,
            )
            if total_ep % 100 == 0:
                print(
                    total_ep,
                    f"time={time.time() - start_time:.4f}",
                    stats.format(
                        "R={R:.4f} r={r:.4f} steps={steps:.4f} success={success:.4f}"
                    ),
                    f"norm={norm:.4f}",
                    f"weight_norm={torch.norm( torch.stack([p.norm(2) for p in policy.parameters()]), 2 ).item():.4f}",
                )

            for step, action, r, adv in zip(
                episode.steps[:-1], episode.actions, returns, advantages, strict=True
            ):
                obs_t.append(torch.tensor(step.observation, dtype=torch.float32))
                returns_t.append(torch.tensor(r, dtype=torch.float32))
                advantages_t.append(torch.tensor(adv, dtype=torch.float))
                actions_t.append(torch.tensor(action.action, dtype=torch.long))
                old_logits_t.append(torch.tensor(action.logits, dtype=torch.float32))

        obs_t, actions_t, returns_t, advantages_t, old_logits_t = (
            torch.stack(obs_t),
            torch.stack(actions_t),
            torch.stack(returns_t),
            torch.stack(advantages_t),
            torch.stack(old_logits_t),
        )

        # returns_t = (returns_t - returns_t.mean()) / (
        #     returns_t.std(unbiased=False) + 1e-8
        # )
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std(unbiased=False) + 1e-8
        )
        old_dist = torch.distributions.Categorical(logits=old_logits_t)
        old_logp: torch.Tensor = old_dist.log_prob(actions_t)
        for _ in range(pi_steps_per_epoch):
            optimizer.zero_grad()
            logits_t = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits_t)
            logp: torch.Tensor = dist.log_prob(actions_t)
            ratio = (logp - old_logp).exp()
            kl = (old_logp - logp).mean().item()
            if kl > 1.5 * 0.01:
                break
            loss = -torch.min(
                ratio * advantages_t,
                torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages_t,
            ).mean()
            # loss = (-ratio * advantages_t).mean()
            # loss = (-logp * advantages_t).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()

        for _ in range(vf_steps_per_epoch):
            vf_optimizer.zero_grad()
            value: torch.Tensor = vf(obs_t)
            loss_v = ((value - returns_t) ** 2).mean()
            loss_v.backward()
            vf_optimizer.step()

    return policy


def main():
    random.seed(0)
    torch.manual_seed(0)

    n = 10
    total_episodes = 50000
    episodes_per_epoch = 100
    epochs = total_episodes // episodes_per_epoch

    env = MazeEnv(
        n=n, fill_prob=0.5, rng=Random(100), max_repeats=1, randomize_agent=False
    )
    val_env = MazeEnv(n=n, fill_prob=0.5, rng=Random(42))
    # policy = MlpPolicy(
    #     input_dim=env.input_dim[0] * env.input_dim[1], action_dim=env.max_action
    # )
    # vf = MlpValue(input_dim=env.input_dim[0] * env.input_dim[1])
    policy = ConvPolicy(
        input_dim=env.input_dim, action_dim=env.max_action, residual=True, ffn_dim=32
    )
    vf = ConvValue(input_dim=env.input_dim)
    # vf = DummyValue()

    optimizer = optim.AdamW(policy.parameters(), lr=3e-4)
    vf_optimizer = optim.AdamW(vf.parameters(), lr=1e-3)
    # optimizer = optim.SGD(policy.parameters(), lr=1e-3, momentum=0.9)
    # vf_optimizer = optim.SGD(vf.parameters(), lr=1e-3, momentum=0.9)
    try:
        policy = ppo_train(
            env=env,
            policy=policy,
            optimizer=optimizer,
            vf=vf,
            vf_optimizer=vf_optimizer,
            epochs=epochs,
            episodes_per_epoch=episodes_per_epoch,
            pi_steps_per_epoch=80,
            vf_steps_per_epoch=80,
            clip_ratio=0.20,
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
        output_dir="maze_output/ppo",
        action_debug_print_fn=lambda x: (
            np.round(x.get("probs", []), 4) if isinstance(x, dict) else None
        ),
    )
    print("Evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
