#!/usr/bin/env python3
"""
06 — DreamerV3 on LunarLander-v3 (Complex Dynamics)
=====================================================
LunarLander is the hardest environment in this project:
- 8-dim observation: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
- 4 discrete actions: noop, fire left, fire main, fire right
- Complex physics: gravity, thrust, legs contact, fuel cost
- Reward: +100~140 for landing, -100 for crash, fuel penalty

Expected: return > 200 within 300-500 episodes.
Requires: pip install gymnasium[box2d]
"""

import gymnasium as gym
import numpy as np
import torch

from dreamer import DreamerAgent, SequenceReplayBuffer, set_seed, get_device
from dreamer import plot_training_curves


def train_dreamer_lunar_lander():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    try:
        env = gym.make('LunarLander-v3')
    except gym.error.DependencyNotInstalled:
        print("LunarLander requires box2d: pip install gymnasium[box2d]")
        print("On macOS you may also need: brew install swig")
        return

    obs_dim = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.n            # 4

    agent = DreamerAgent(
        obs_dim=obs_dim, action_dim=action_dim, discrete=True, device=device,
        gru_dim=256, stoch_dim=16, stoch_classes=16,
        horizon=15, gamma=0.99, lam=0.95,
        lr_world=3e-4, lr_actor=1e-4, lr_critic=1e-4,
        entropy_coef=3e-4, kl_coef=0.5,
    )

    buffer = SequenceReplayBuffer(max_episodes=1000)

    # ── Hyperparameters ──
    num_episodes = 500
    prefill_steps = 3000
    warmup_steps = 300
    train_every = 5
    batch_size = 16
    seq_len = 32

    # ── Prefill ──
    print(f"Prefilling buffer with {prefill_steps} random steps...")
    obs, _ = env.reset()
    steps = 0
    while steps < prefill_steps:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs.astype(np.float32),
                   np.array([action], dtype=np.float32),
                   float(reward), done)
        obs = next_obs if not done else env.reset()[0]
        steps += 1
    print(f"  Buffer: {len(buffer)} episodes, {buffer.total_steps} steps")

    # ── World model warmup ──
    print(f"\nWarming up world model for {warmup_steps} steps...")
    for step in range(warmup_steps):
        batch = buffer.sample(batch_size, seq_len, device)
        losses = agent.train_step(batch, world_model_only=True)
        if (step + 1) % 100 == 0:
            print(f"  Warmup {step+1}/{warmup_steps} | "
                  f"world={losses['world_total']:.4f}")

    # ── Training ──
    episode_returns = []
    loss_history = {'world_total': [], 'actor': [], 'critic': [], 'kl': []}
    global_step = 0
    best_avg = -float('inf')

    print(f"\nTraining DreamerV3 on LunarLander-v3 for {num_episodes} episodes...")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        agent.reset_state()
        ep_return = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(obs.astype(np.float32),
                       np.array([action], dtype=np.float32),
                       float(reward), done)

            global_step += 1
            if global_step % train_every == 0 and len(buffer) > 0:
                batch = buffer.sample(batch_size, seq_len, device)
                losses = agent.train_step(batch)
                for k in loss_history:
                    if k in losses:
                        loss_history[k].append(losses[k])

            obs = next_obs
            ep_return += reward

        episode_returns.append(ep_return)

        if ep % 25 == 0 or ep == 1:
            recent = np.mean(episode_returns[-25:])
            best_avg = max(best_avg, recent)
            print(f"  Episode {ep:3d} | Return: {ep_return:7.1f} | "
                  f"Avg25: {recent:7.1f} | BestAvg: {best_avg:7.1f}")

        if len(episode_returns) >= 50 and np.mean(episode_returns[-50:]) > 200:
            print(f"\n  ★ Solved! 50-ep average: {np.mean(episode_returns[-50:]):.1f}")
            break

    env.close()

    plot_training_curves(
        rewards=episode_returns,
        losses=loss_history,
        title='DreamerV3 LunarLander-v3',
        save_path='assets/06_lunar_lander_training.png'
    )

    torch.save(agent.state_dict(), 'assets/dreamer_lunar_lander.pt')
    print(f"\nFinal 25-ep avg: {np.mean(episode_returns[-25:]):.1f}")
    print(f"Model saved to assets/dreamer_lunar_lander.pt")


if __name__ == '__main__':
    train_dreamer_lunar_lander()
