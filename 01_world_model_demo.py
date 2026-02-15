#!/usr/bin/env python3
"""
01 — World Model Training Demo
================================
Train ONLY the world model (RSSM + decoder + reward/continue heads)
on random CartPole rollouts. No actor-critic yet — just verifying
that the world model can learn environment dynamics.

What to expect:
- Decoder loss decreases (model learns to reconstruct observations)
- Reward/continue losses decrease (model predicts episode signals)
- KL stays above free bits (prior learns meaningful representations)
- Prediction visualization shows model tracking real trajectories
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer import (
    set_seed, get_device, RSSM, Encoder, Decoder, RewardHead, ContinueHead,
    SequenceReplayBuffer, plot_training_curves, plot_world_model_predictions,
    symlog, symexp, twohot_decode
)


def collect_random_episodes(env_name='CartPole-v1', num_episodes=50):
    """Collect episodes using random policy."""
    env = gym.make(env_name)
    buffer = SequenceReplayBuffer(max_episodes=1000)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs.astype(np.float32),
                       np.array([action], dtype=np.float32),
                       float(reward), done)
            obs = next_obs

    env.close()
    print(f"Collected {num_episodes} episodes, {buffer.total_steps} total steps")
    return buffer


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # ── Collect data ──
    buffer = collect_random_episodes('CartPole-v1', num_episodes=50)

    # ── Build world model ──
    obs_dim = 4       # CartPole observation: [x, x_dot, theta, theta_dot]
    action_dim = 2    # CartPole actions: left, right
    embed_dim = 256
    gru_dim = 256

    encoder = Encoder(obs_dim, embed_dim).to(device)
    rssm = RSSM(obs_dim, action_dim, embed_dim, gru_dim).to(device)
    decoder = Decoder(rssm.latent_dim, obs_dim).to(device)
    reward_head = RewardHead(rssm.latent_dim).to(device)
    continue_head = ContinueHead(rssm.latent_dim).to(device)

    params = (list(encoder.parameters()) + list(rssm.parameters()) +
              list(decoder.parameters()) + list(reward_head.parameters()) +
              list(continue_head.parameters()))
    optimizer = torch.optim.Adam(params, lr=1e-4)

    # ── Training ──
    num_steps = 200
    batch_size = 16
    seq_len = 32
    losses_log = {'decoder': [], 'reward': [], 'continue': [], 'kl': []}

    print(f"\nTraining world model for {num_steps} steps...")

    for step in range(1, num_steps + 1):
        batch = buffer.sample(batch_size, seq_len, device)
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        mask = batch['mask']

        B, T = obs.shape[:2]

        # Encode
        embeds = encoder(obs)

        # Convert actions to one-hot
        act_idx = actions.long().squeeze(-1)
        act_onehot = F.one_hot(act_idx, action_dim).float()

        # RSSM forward
        h_init, z_init = rssm.initial_state(B, device)
        h_states, z_posts, prior_probs, post_probs = rssm.observe_sequence(
            embeds, act_onehot, h_init, z_init
        )

        latents = rssm.get_latent(h_states, z_posts)

        # Mask valid (non-padded) steps
        mask_flat = mask.reshape(B * T)
        valid_idx = mask_flat > 0.5
        latents_valid = latents.reshape(B * T, -1)[valid_idx]
        obs_valid = obs.reshape(B * T, -1)[valid_idx]
        rewards_valid = rewards.reshape(B * T)[valid_idx]
        continues_valid = (1 - dones).reshape(B * T)[valid_idx]

        # Losses
        decoder_loss = decoder.loss(latents_valid, obs_valid)
        reward_loss = reward_head.loss(latents_valid, rewards_valid)
        continue_loss = continue_head.loss(latents_valid, continues_valid)
        kl_loss = rssm.kl_loss(prior_probs, post_probs, free_bits=1.0)

        total = decoder_loss + reward_loss + continue_loss + 0.5 * kl_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()

        for k, v in [('decoder', decoder_loss), ('reward', reward_loss),
                      ('continue', continue_loss), ('kl', kl_loss)]:
            losses_log[k].append(v.item())

        if step % 50 == 0 or step == 1:
            print(f"  Step {step:3d} | dec={decoder_loss.item():.4f} "
                  f"rew={reward_loss.item():.4f} cont={continue_loss.item():.4f} "
                  f"kl={kl_loss.item():.4f}")

    # ── Visualize predictions ──
    print("\nGenerating prediction visualizations...")

    # Get a test episode
    env = gym.make('CartPole-v1')
    obs, _ = env.reset()
    test_obs, test_actions, test_rewards = [], [], []
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        test_obs.append(obs.astype(np.float32))
        test_actions.append([action])
        test_rewards.append(reward)
        obs = next_obs
    env.close()

    test_obs = np.array(test_obs)
    test_actions = np.array(test_actions)
    test_rewards = np.array(test_rewards)

    # Predict
    with torch.no_grad():
        obs_t = torch.tensor(test_obs, device=device).unsqueeze(0)
        act_t = F.one_hot(torch.tensor(test_actions, device=device).long().squeeze(-1),
                         action_dim).float().unsqueeze(0)
        embeds = encoder(obs_t)
        h_init, z_init = rssm.initial_state(1, device)
        h_states, z_posts, _, _ = rssm.observe_sequence(embeds, act_t, h_init, z_init)
        latents = rssm.get_latent(h_states, z_posts).reshape(-1, rssm.latent_dim)

        pred_obs = symexp(decoder(latents)).cpu().numpy()
        pred_rewards = twohot_decode(reward_head(latents)).cpu().numpy()

    # Plot
    plot_training_curves(
        rewards=[],  # No reward curve for world model only
        losses=losses_log,
        title='World Model',
        save_path='assets/01_world_model_losses.png'
    )

    plot_world_model_predictions(
        real_obs=test_obs, pred_obs=pred_obs,
        real_rewards=test_rewards, pred_rewards=pred_rewards,
        title='World Model',
        save_path='assets/01_world_model_predictions.png'
    )

    print("\n✓ World model demo complete!")
    print(f"  Final losses — decoder: {losses_log['decoder'][-1]:.4f}, "
          f"reward: {losses_log['reward'][-1]:.4f}, kl: {losses_log['kl'][-1]:.4f}")


if __name__ == '__main__':
    main()
