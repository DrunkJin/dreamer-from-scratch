#!/usr/bin/env python3
"""
03 — DreamerV3 on Pendulum-v1 (Continuous Actions)
====================================================
Continuous control test — the actor outputs mean+std for a Normal
distribution, squashed through tanh to [-2, 2] torque range.

Pendulum is harder than CartPole because:
- Continuous action space (infinite possible actions)
- Must learn to swing up AND balance (two-phase behavior)
- Dense negative rewards (no sparse reward signal)

Expected: return > -300 within 150-250 episodes.
"""

import gymnasium as gym
import numpy as np
import torch

from dreamer import DreamerAgent, SequenceReplayBuffer, set_seed, get_device
from dreamer import plot_training_curves


def train_dreamer_pendulum():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0]   # 3: [cos(theta), sin(theta), theta_dot]
    action_dim = env.action_space.shape[0]      # 1: torque

    agent = DreamerAgent(
        obs_dim=obs_dim, action_dim=action_dim, discrete=False, device=device,
        gru_dim=256, stoch_dim=16, stoch_classes=16,
        horizon=15, gamma=0.99, lam=0.95,
        lr_world=3e-4, lr_actor=1e-4, lr_critic=1e-4,
        entropy_coef=1e-4, kl_coef=0.5,
    )

    buffer = SequenceReplayBuffer(max_episodes=500)

    # ── Hyperparameters ──
    num_episodes = 100
    prefill_steps = 2000
    warmup_steps = 200
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
        buffer.add(obs.astype(np.float32), action.astype(np.float32),
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
                  f"world={losses['world_total']:.4f} kl={losses['kl']:.4f}")

    # ── Training ──
    episode_returns = []
    loss_history = {'world_total': [], 'actor': [], 'critic': [], 'kl': []}
    global_step = 0

    print(f"\nTraining DreamerV3 on Pendulum-v1 for {num_episodes} episodes...")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        agent.reset_state()
        ep_return = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            # Scale action from [-1, 1] to environment range [-2, 2]
            env_action = np.clip(action * 2.0, -2.0, 2.0)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            buffer.add(obs.astype(np.float32), action.astype(np.float32),
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

        if ep % 20 == 0 or ep == 1:
            recent = np.mean(episode_returns[-20:])
            print(f"  Episode {ep:3d} | Return: {ep_return:7.1f} | "
                  f"Avg20: {recent:7.1f}")

        # Check if solved
        if len(episode_returns) >= 30 and np.mean(episode_returns[-30:]) > -300:
            print(f"\n  ★ Solved! 30-episode average: {np.mean(episode_returns[-30:]):.1f}")
            break

    env.close()

    plot_training_curves(
        rewards=episode_returns,
        losses=loss_history,
        title='DreamerV3 Pendulum-v1',
        save_path='assets/03_pendulum_training.png'
    )

    torch.save(agent.state_dict(), 'assets/dreamer_pendulum.pt')
    print(f"\nFinal 20-ep avg: {np.mean(episode_returns[-20:]):.1f}")
    print(f"Model saved to assets/dreamer_pendulum.pt")


if __name__ == '__main__':
    train_dreamer_pendulum()
