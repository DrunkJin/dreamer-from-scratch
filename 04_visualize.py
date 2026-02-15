#!/usr/bin/env python3
"""
04 — Visualization: What Does the World Model Learn?
=====================================================
After training, we can peer inside DreamerV3's learned world model:

1. Imagined vs Real trajectories — does the model predict accurately?
2. Latent space t-SNE — do similar states cluster together?
3. Reward prediction accuracy — scatter plot of predicted vs real rewards
4. Imagination rollout — what does the agent "see" when dreaming?
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

from dreamer import DreamerAgent, set_seed, get_device
from dreamer.utils import symexp, twohot_decode


def collect_evaluation_data(env, agent, num_episodes=10):
    """Collect real trajectories with the trained agent."""
    all_obs, all_actions, all_rewards, all_latents = [], [], [], []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        agent.reset_state()
        agent.init_state(1)

        ep_obs, ep_actions, ep_rewards = [], [], []
        done = False

        while not done:
            ep_obs.append(obs.astype(np.float32))
            action = agent.select_action(obs, explore=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if agent.discrete:
                ep_actions.append([action])
            else:
                ep_actions.append(action)
            ep_rewards.append(reward)
            obs = next_obs

        all_obs.append(np.array(ep_obs))
        all_actions.append(np.array(ep_actions))
        all_rewards.append(np.array(ep_rewards))

    return all_obs, all_actions, all_rewards


def plot_imagined_vs_real(agent, obs_seq, action_seq, reward_seq, save_path,
                          env_name='Environment'):
    """Compare imagined trajectory with real trajectory."""
    pred_obs, pred_rewards = agent.predict_sequence(obs_seq, action_seq)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    T = min(len(obs_seq), len(pred_obs))
    ndim = min(4, obs_seq.shape[-1])

    # Obs dims
    for d in range(min(2, ndim)):
        ax = axes[0, d]
        ax.plot(obs_seq[:T, d], label='Real', linewidth=2, alpha=0.8)
        ax.plot(pred_obs[:T, d], '--', label='Imagined', linewidth=2, alpha=0.8)
        ax.set_title(f'Observation Dim {d}')
        ax.set_xlabel('Time Step')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Reward comparison
    T_r = min(len(reward_seq), len(pred_rewards))
    axes[1, 0].plot(reward_seq[:T_r], label='Real', linewidth=2, alpha=0.8)
    axes[1, 0].plot(pred_rewards[:T_r], '--', label='Imagined', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Reward Prediction')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Prediction error over time
    if ndim > 0:
        obs_error = np.mean((obs_seq[:T] - pred_obs[:T]) ** 2, axis=-1)
        axes[1, 1].plot(obs_error, color='coral', linewidth=2)
        axes[1, 1].set_title('Observation MSE Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Imagined vs Real — {env_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_latent_tsne(agent, all_obs, all_actions, all_rewards, save_path,
                     env_name='Environment'):
    """t-SNE visualization of learned latent space."""
    device = agent.device
    latents_list = []
    rewards_list = []

    for obs_seq, action_seq, reward_seq in zip(all_obs, all_actions, all_rewards):
        with torch.no_grad():
            obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)
            act_t = torch.tensor(action_seq, dtype=torch.float32, device=device).unsqueeze(0)

            if agent.discrete and act_t.shape[-1] != agent.action_dim:
                act_t = F.one_hot(act_t.long().squeeze(-1), agent.action_dim).float()

            embeds = agent.encoder(obs_t)
            h_init, z_init = agent.rssm.initial_state(1, device)
            h_states, z_posts, _, _ = agent.rssm.observe_sequence(
                embeds, act_t, h_init, z_init
            )
            latent = agent.rssm.get_latent(h_states, z_posts).squeeze(0)
            latents_list.append(latent.cpu().numpy())
            rewards_list.append(reward_seq)

    all_latents = np.concatenate(latents_list, axis=0)
    all_rewards_flat = np.concatenate(rewards_list, axis=0)

    # Subsample if too many points
    max_points = 2000
    if len(all_latents) > max_points:
        idx = np.random.choice(len(all_latents), max_points, replace=False)
        all_latents = all_latents[idx]
        all_rewards_flat = all_rewards_flat[idx]

    # t-SNE
    print("  Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    latents_2d = tsne.fit_transform(all_latents)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color by reward
    sc = axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1],
                         c=all_rewards_flat, cmap='RdYlGn', s=8, alpha=0.6)
    plt.colorbar(sc, ax=axes[0], label='Reward')
    axes[0].set_title('Latent Space — Colored by Reward')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    # Color by time step within episode
    time_colors = []
    for reward_seq in rewards_list:
        t = np.arange(len(reward_seq)) / max(1, len(reward_seq) - 1)
        time_colors.append(t)
    all_times = np.concatenate(time_colors)
    if len(all_times) > max_points:
        all_times = all_times[idx]

    sc2 = axes[1].scatter(latents_2d[:, 0], latents_2d[:, 1],
                          c=all_times, cmap='viridis', s=8, alpha=0.6)
    plt.colorbar(sc2, ax=axes[1], label='Episode Progress')
    axes[1].set_title('Latent Space — Colored by Time')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.suptitle(f'Latent Space t-SNE — {env_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_reward_scatter(agent, all_obs, all_actions, all_rewards, save_path,
                        env_name='Environment'):
    """Scatter plot: predicted reward vs actual reward."""
    device = agent.device
    real_all, pred_all = [], []

    for obs_seq, action_seq, reward_seq in zip(all_obs, all_actions, all_rewards):
        pred_obs, pred_rewards = agent.predict_sequence(obs_seq, action_seq)
        T = min(len(reward_seq), len(pred_rewards))
        real_all.extend(reward_seq[:T])
        pred_all.extend(pred_rewards[:T])

    real_all = np.array(real_all)
    pred_all = np.array(pred_all)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(real_all, pred_all, alpha=0.3, s=10, color='steelblue')

    # Perfect prediction line
    lims = [min(real_all.min(), pred_all.min()),
            max(real_all.max(), pred_all.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')

    # Correlation
    corr = np.corrcoef(real_all, pred_all)[0, 1]
    ax.set_title(f'Reward Prediction Accuracy — {env_name}\n(r = {corr:.3f})')
    ax.set_xlabel('Real Reward')
    ax.set_ylabel('Predicted Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_environment(env_name, agent_path, discrete, obs_dim, action_dim,
                          prefix='04'):
    """Full visualization pipeline for one environment."""
    device = get_device()

    agent = DreamerAgent(
        obs_dim=obs_dim, action_dim=action_dim, discrete=discrete, device=device,
    )

    if not os.path.exists(agent_path):
        print(f"  Skipping {env_name}: model not found at {agent_path}")
        return

    agent.load_state_dict(torch.load(agent_path, map_location=device, weights_only=True))
    agent.eval()

    env = gym.make(env_name)
    print(f"\nVisualizing {env_name}...")

    all_obs, all_actions, all_rewards = collect_evaluation_data(env, agent, num_episodes=10)

    # Pick longest episode for trajectory comparison
    longest_idx = max(range(len(all_obs)), key=lambda i: len(all_obs[i]))

    plot_imagined_vs_real(
        agent, all_obs[longest_idx], all_actions[longest_idx], all_rewards[longest_idx],
        f'assets/{prefix}_{env_name.lower().replace("-", "_")}_imagined_vs_real.png',
        env_name
    )

    plot_latent_tsne(
        agent, all_obs, all_actions, all_rewards,
        f'assets/{prefix}_{env_name.lower().replace("-", "_")}_latent_tsne.png',
        env_name
    )

    plot_reward_scatter(
        agent, all_obs, all_actions, all_rewards,
        f'assets/{prefix}_{env_name.lower().replace("-", "_")}_reward_scatter.png',
        env_name
    )

    env.close()


def main():
    set_seed(42)

    print("=" * 60)
    print("DreamerV3 Visualization Suite")
    print("=" * 60)

    # CartPole
    visualize_environment(
        'CartPole-v1', 'assets/dreamer_cartpole.pt',
        discrete=True, obs_dim=4, action_dim=2, prefix='04'
    )

    # Pendulum
    visualize_environment(
        'Pendulum-v1', 'assets/dreamer_pendulum.pt',
        discrete=False, obs_dim=3, action_dim=1, prefix='04'
    )

    # Acrobot
    visualize_environment(
        'Acrobot-v1', 'assets/dreamer_acrobot.pt',
        discrete=True, obs_dim=6, action_dim=3, prefix='04'
    )

    # LunarLander (optional)
    try:
        visualize_environment(
            'LunarLander-v3', 'assets/dreamer_lunar_lander.pt',
            discrete=True, obs_dim=8, action_dim=4, prefix='04'
        )
    except Exception as e:
        print(f"  LunarLander visualization skipped: {e}")

    print("\n✓ All visualizations complete!")


if __name__ == '__main__':
    main()
