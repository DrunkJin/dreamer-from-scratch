#!/usr/bin/env python3
"""
02 — DreamerV3 on CartPole-v1 (Discrete Actions)
==================================================
Full DreamerV3 training loop on CartPole — the simplest test case
for validating that world model + actor-critic imagination works.

Expected: solve (return > 450) within 100-200 episodes.

Training phases:
  1. Prefill: collect random experience
  2. Warmup: train world model only (so imagination is meaningful)
  3. Full training: world model + actor-critic in imagination
"""

import gymnasium as gym
import numpy as np
import torch

from dreamer import DreamerAgent, SequenceReplayBuffer, set_seed, get_device
from dreamer import plot_training_curves


def train_dreamer_cartpole():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n            # 2

    agent = DreamerAgent(
        obs_dim=obs_dim, action_dim=action_dim, discrete=True, device=device,
        gru_dim=256, stoch_dim=16, stoch_classes=16,
        horizon=15, gamma=0.99, lam=0.95,
        lr_world=3e-4, lr_actor=1e-4, lr_critic=1e-4,
        entropy_coef=1e-3, kl_coef=0.5,
    )

    buffer = SequenceReplayBuffer(max_episodes=500)

    # ── Hyperparameters ──
    num_episodes = 200
    prefill_steps = 1000
    warmup_steps = 500       # World model warmup (needs enough to learn continue)
    train_every = 5
    batch_size = 16
    seq_len = 32

    # ── Prefill with random actions ──
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
        if (step + 1) % 50 == 0:
            print(f"  Warmup {step+1}/{warmup_steps} | "
                  f"world={losses['world_total']:.4f} kl={losses['kl']:.4f}")

    # ── Training loop ──
    episode_returns = []
    loss_history = {'world_total': [], 'actor': [], 'critic': [], 'kl': []}
    global_step = 0
    best_return = -float('inf')

    print(f"\nTraining DreamerV3 on CartPole-v1 for {num_episodes} episodes...")

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

            # Train every N steps
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
        best_return = max(best_return, ep_return)

        if ep % 10 == 0 or ep == 1:
            recent = np.mean(episode_returns[-10:])
            print(f"  Episode {ep:3d} | Return: {ep_return:6.1f} | "
                  f"Avg10: {recent:6.1f} | Best: {best_return:6.1f} | "
                  f"Steps: {global_step}")

        # Early success check
        if len(episode_returns) >= 20 and np.mean(episode_returns[-20:]) > 450:
            print(f"\n  ★ Solved! 20-episode average: {np.mean(episode_returns[-20:]):.1f}")
            break

    env.close()

    # ── Plot ──
    plot_training_curves(
        rewards=episode_returns,
        losses=loss_history,
        title='DreamerV3 CartPole-v1',
        save_path='assets/02_cartpole_training.png'
    )

    # Save model
    torch.save(agent.state_dict(), 'assets/dreamer_cartpole.pt')
    print(f"\nFinal 20-ep avg: {np.mean(episode_returns[-20:]):.1f}")
    print(f"Model saved to assets/dreamer_cartpole.pt")


if __name__ == '__main__':
    train_dreamer_cartpole()
