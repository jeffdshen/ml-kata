import random
import time
from pathlib import Path
from random import Random

import torch
import torch.optim as optim

from kata12.maze_env import *
from kata12.models import *
from kata12.types import *


def grpo_train(
    env: Env2D,
    policy: Policy,
    optimizer: optim.Optimizer,
    epochs: int,
    episodes_per_epoch: int,
    group_size: int,
    pi_steps_per_epoch: int,
    clip_ratio: float,
    kl_beta: float,
    obs_to_env: Callable[[list[int]], Env2D],
):
    stats = EmaStats()
    start_time = time.time()

    for epoch in range(epochs):
        obs_t = []
        actions_t = []
        old_logits_t = []
        advantages_t = []

        for group_num in range(episodes_per_epoch // group_size):
            group_env = obs_to_env(env.reset().observation)
            group_episodes = [group_env.run_episode(policy) for _ in range(group_size)]
            group_returns = [episode.total_return for episode in group_episodes]
            mean_r, std_r = np.mean(group_returns), np.std(group_returns)

            for ep_num, (ep, r) in enumerate(zip(group_episodes, group_returns)):
                total_ep = epoch * episodes_per_epoch + group_num * group_size + ep_num
                adv = (r - mean_r) / (std_r + 1e-8)
                stats.log_ema(
                    {
                        "R": ep.total_return,
                        "steps": len(ep.steps),
                        "success": ep.success,
                    },
                    ratio=0.99,
                )
                if total_ep % 100 == 0:
                    print(
                        total_ep,
                        f"time={time.time() - start_time:.4f}",
                        stats.format(
                            "R={R:.4f} steps={steps:.4f} success={success:.4f}"
                        ),
                        f"weight_norm={torch.norm( torch.stack([p.norm(2) for p in policy.parameters()]), 2 ).item():.4f}",
                    )

                for step, action in zip(ep.steps[:-1], ep.actions, strict=True):
                    obs_t.append(torch.tensor(step.observation, dtype=torch.float32))
                    advantages_t.append(torch.tensor(adv, dtype=torch.float))
                    actions_t.append(torch.tensor(action.action, dtype=torch.long))
                    old_logits_t.append(
                        torch.tensor(action.logits, dtype=torch.float32)
                    )

        obs_t, actions_t, advantages_t, old_logits_t = (
            torch.stack(obs_t),
            torch.stack(actions_t),
            torch.stack(advantages_t),
            torch.stack(old_logits_t),
        )

        old_dist = torch.distributions.Categorical(logits=old_logits_t)
        old_logp: torch.Tensor = old_dist.log_prob(actions_t)
        for _ in range(pi_steps_per_epoch):
            optimizer.zero_grad()
            logits_t = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits_t)
            logp: torch.Tensor = dist.log_prob(actions_t)
            ratio = (logp - old_logp).exp()
            loss = -torch.min(
                ratio * advantages_t,
                torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages_t,
            ).mean()
            # TODO: should actually be not pi_old but pi_ref, which is updated less often.
            loss += kl_beta * (1.0 / ratio + logp - old_logp - 1).mean()
            loss.backward()
            optimizer.step()

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
    policy = ConvPolicy(
        input_dim=env.input_dim, action_dim=env.max_action, residual=True, ffn_dim=32
    )

    optimizer = optim.AdamW(policy.parameters(), lr=3e-4)
    # optimizer = optim.SGD(policy.parameters(), lr=1e-3, momentum=0.9)
    try:
        policy = grpo_train(
            env=env,
            policy=policy,
            optimizer=optimizer,
            epochs=epochs,
            episodes_per_epoch=episodes_per_epoch,
            group_size=10,
            pi_steps_per_epoch=10,
            clip_ratio=0.20,
            kl_beta=0.01,
            obs_to_env=lambda x: maze_obs_to_env(x, n, env.max_steps),
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
        output_dir="maze_output/grpo",
        action_debug_print_fn=lambda x: (
            np.round(x.get("probs", []), 4) if isinstance(x, dict) else None
        ),
    )
    print("Evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
