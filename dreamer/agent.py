"""
DreamerAgent — The Full DreamerV3 Algorithm
=============================================
Ties together the world model (RSSM + prediction heads) and the
actor-critic, orchestrating the two-phase learning process:

Phase 1: WORLD MODEL LEARNING (from real experience)
  - Encode observations, run RSSM to get posterior states
  - Train decoder (reconstruct obs), reward head, continue head
  - Minimize KL divergence between posterior and prior

Phase 2: BEHAVIOR LEARNING (in imagination)
  - Start from posterior states, dream forward using actor + prior
  - Compute lambda-returns from imagined rewards and values
  - Update actor to maximize returns, critic to predict returns
  - No environment interaction needed!

This separation is the key insight: the world model compresses
experience into a compact representation, and the actor learns
from unlimited imagined experience within that representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .rssm import RSSM
from .networks import Encoder, Decoder, RewardHead, ContinueHead, Actor, Critic
from .utils import symlog, twohot_decode, ReturnNormalizer


class DreamerAgent(nn.Module):
    """Complete DreamerV3 agent."""

    def __init__(self, obs_dim, action_dim, discrete=True, device='cpu',
                 gru_dim=256, stoch_dim=16, stoch_classes=16,
                 embed_dim=256, hidden=256, num_bins=255,
                 horizon=15, gamma=0.99, lam=0.95,
                 lr_world=1e-4, lr_actor=3e-5, lr_critic=3e-5,
                 entropy_coef=3e-4, free_bits=1.0, kl_coef=0.5):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = device
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.free_bits = free_bits
        self.kl_coef = kl_coef
        self.num_bins = num_bins

        # ── World Model ──
        self.encoder = Encoder(obs_dim, embed_dim, hidden).to(device)
        self.rssm = RSSM(obs_dim, action_dim, embed_dim,
                         gru_dim, stoch_dim, stoch_classes,
                         hidden).to(device)
        self.decoder = Decoder(self.rssm.latent_dim, obs_dim, hidden).to(device)
        self.reward_head = RewardHead(self.rssm.latent_dim, num_bins, hidden).to(device)
        self.continue_head = ContinueHead(self.rssm.latent_dim, hidden).to(device)

        # ── Actor-Critic ──
        self.actor = Actor(self.rssm.latent_dim, action_dim, discrete, hidden).to(device)
        self.critic = Critic(self.rssm.latent_dim, num_bins, hidden).to(device)

        # Slow critic target for stable value estimation
        self.critic_target = Critic(self.rssm.latent_dim, num_bins, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Optimizers ──
        world_params = (list(self.encoder.parameters()) +
                       list(self.rssm.parameters()) +
                       list(self.decoder.parameters()) +
                       list(self.reward_head.parameters()) +
                       list(self.continue_head.parameters()))
        self.world_optimizer = torch.optim.Adam(world_params, lr=lr_world, eps=1e-5)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        # Return normalization
        self.return_normalizer = ReturnNormalizer()

        # Training state
        self._h = None
        self._z = None

    def init_state(self, batch_size=1):
        """Initialize RSSM state for environment interaction."""
        self._h, self._z = self.rssm.initial_state(batch_size, self.device)
        return self._h, self._z

    @torch.no_grad()
    def select_action(self, obs, explore=True):
        """Select action for environment interaction.

        Uses the posterior (observe_step with real obs) to maintain
        an accurate internal state, then samples from the actor.
        """
        if self._h is None:
            self.init_state(1)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        embed = self.encoder(obs_t)

        # Dummy action for first step (doesn't matter, GRU handles it)
        if not hasattr(self, '_last_action'):
            self._last_action = torch.zeros(1, self.action_dim, device=self.device)

        self._h, self._z, _, _ = self.rssm.observe_step(
            self._h, self._z, self._last_action, embed
        )

        latent = self.rssm.get_latent(self._h, self._z)

        if self.discrete:
            action_idx, action_vec, dist = self.actor.sample(latent)
            action = action_idx.item()
            self._last_action = F.one_hot(
                action_idx, self.action_dim
            ).float()
        else:
            action_cont, _, dist = self.actor.sample(latent)
            action = action_cont.squeeze(0).cpu().numpy()
            self._last_action = action_cont

        return action

    def reset_state(self):
        """Reset RSSM state at episode boundary."""
        self._h = None
        self._z = None
        if hasattr(self, '_last_action'):
            del self._last_action

    def train_step(self, batch, world_model_only=False):
        """One full training step: world model + actor-critic.

        Args:
            batch: Sampled sequence batch from replay buffer
            world_model_only: If True, only train world model (warmup phase)

        Returns dict of loss values for logging.
        """
        wm_losses = self._train_world_model(batch)

        if world_model_only:
            return wm_losses

        ac_losses = self._train_actor_critic(batch)

        # Soft-update critic target
        self._update_target(tau=0.02)

        return {**wm_losses, **ac_losses}

    def _train_world_model(self, batch):
        """Phase 1: Train world model on real experience sequences."""
        obs = batch['obs']        # (B, T, obs_dim)
        actions = batch['actions']  # (B, T, action_dim)
        rewards = batch['rewards']  # (B, T)
        dones = batch['dones']    # (B, T)
        mask = batch['mask']      # (B, T) — 1.0 for valid, 0.0 for padded

        B, T = obs.shape[:2]

        # Encode observations
        embeds = self.encoder(obs)  # (B, T, embed_dim)

        # Prepare action tensors
        if self.discrete:
            if actions.dim() == 2:
                actions = actions.long()
                actions = F.one_hot(actions, self.action_dim).float()
            elif actions.shape[-1] != self.action_dim:
                actions = actions.long().squeeze(-1)
                actions = F.one_hot(actions, self.action_dim).float()

        # Initialize RSSM state
        h_init, z_init = self.rssm.initial_state(B, self.device)

        # Run RSSM over sequence
        h_states, z_posts, prior_probs, post_probs = self.rssm.observe_sequence(
            embeds, actions, h_init, z_init
        )

        # Compute latent states
        latents = self.rssm.get_latent(h_states, z_posts)  # (B, T, latent_dim)

        # Apply mask: only compute losses on valid (non-padded) steps
        mask_flat = mask.reshape(B * T)
        valid_idx = mask_flat > 0.5
        latents_valid = latents.reshape(B * T, -1)[valid_idx]
        obs_valid = obs.reshape(B * T, -1)[valid_idx]
        rewards_valid = rewards.reshape(B * T)[valid_idx]

        continues = 1.0 - dones
        continues_valid = continues.reshape(B * T)[valid_idx]

        # ── Losses (only on valid steps) ──
        decoder_loss = self.decoder.loss(latents_valid, obs_valid)
        reward_loss = self.reward_head.loss(latents_valid, rewards_valid)
        continue_loss = self.continue_head.loss(latents_valid, continues_valid)

        # KL divergence (masked)
        kl_loss = self.rssm.kl_loss(prior_probs, post_probs, self.free_bits)

        total_loss = decoder_loss + reward_loss + continue_loss + self.kl_coef * kl_loss

        self.world_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.rssm.parameters()) +
            list(self.decoder.parameters()) + list(self.reward_head.parameters()) +
            list(self.continue_head.parameters()),
            10.0
        )
        self.world_optimizer.step()

        return {
            'decoder': decoder_loss.item(),
            'reward': reward_loss.item(),
            'continue': continue_loss.item(),
            'kl': kl_loss.item(),
            'world_total': total_loss.item(),
        }

    def _train_actor_critic(self, batch):
        """Phase 2: Train actor-critic in imagination.

        Key insight: gradients flow through the world model dynamics
        to the actor. The world model parameters are frozen (not in
        actor optimizer), but the computation graph connects:
        actor(latent) → action → imagine_step → next_latent → reward

        This lets the actor learn "if I take this action, the world
        model predicts higher reward" — much more efficient than
        pure REINFORCE which only gets scalar reward signal.

        For discrete actions, straight-through gradients + REINFORCE
        are combined for robust learning.
        """
        obs = batch['obs']
        actions = batch['actions']
        B, T = obs.shape[:2]

        # Get posterior states (detach from world model graph)
        with torch.no_grad():
            embeds = self.encoder(obs)
            if self.discrete:
                if actions.dim() == 2:
                    act = F.one_hot(actions.long(), self.action_dim).float()
                elif actions.shape[-1] != self.action_dim:
                    act = F.one_hot(actions.long().squeeze(-1), self.action_dim).float()
                else:
                    act = actions
            else:
                act = actions

            h_init, z_init = self.rssm.initial_state(B, self.device)
            h_states, z_posts, _, _ = self.rssm.observe_sequence(
                embeds, act, h_init, z_init
            )

        # Use valid posterior states as imagination starting points
        mask = batch['mask']  # (B, T)
        mask_flat = mask.reshape(B * T)
        valid_idx = mask_flat > 0.5

        h_flat = h_states.reshape(B * T, -1)[valid_idx].detach()
        z_flat = z_posts.reshape(B * T, *z_posts.shape[2:])[valid_idx].detach()
        N = h_flat.shape[0]  # Number of valid imagination trajectories

        # ── Imagine forward (gradients flow through actor → dynamics) ──
        imagined_latents = []
        imagined_entropies = []

        h, z = h_flat, z_flat

        for _ in range(self.horizon):
            latent = self.rssm.get_latent(h, z)
            imagined_latents.append(latent)

            # Actor selects action (gradient flows through here)
            if self.discrete:
                action_idx, action_vec, dist = self.actor.sample(latent)
                entropy = dist.entropy()
                action = action_vec  # Straight-through gradient
            else:
                action_cont, _, dist = self.actor.sample(latent)
                entropy = dist.entropy().sum(-1)
                action = action_cont  # Reparameterized gradient

            imagined_entropies.append(entropy)

            # World model imagines next state — gradient flows through action
            h, z = self.rssm.imagine_step(h, z, action)

        # Final state value
        final_latent = self.rssm.get_latent(h, z)
        imagined_latents.append(final_latent)

        latents_stack = torch.stack(imagined_latents, dim=1)  # (N, H+1, latent_dim)
        flat_latents = latents_stack[:, :-1].reshape(-1, latents_stack.shape[-1])

        # Predict rewards from imagined states (gradient flows to actor!)
        reward_logits = self.reward_head(flat_latents)
        imagined_rewards = twohot_decode(reward_logits, self.num_bins)
        imagined_rewards = imagined_rewards.reshape(N, self.horizon)

        continue_logits = self.continue_head(flat_latents)
        imagined_continues = torch.sigmoid(continue_logits).reshape(N, self.horizon)

        # Values from target critic (no gradient — just for bootstrapping)
        with torch.no_grad():
            value_logits = self.critic_target(
                latents_stack.reshape(-1, latents_stack.shape[-1])
            )
            imagined_values = twohot_decode(value_logits, self.num_bins)
            imagined_values = imagined_values.reshape(N, self.horizon + 1)

        # ── Compute lambda-returns ──
        lambda_returns = self._compute_lambda_returns(
            imagined_rewards, imagined_values, imagined_continues
        )

        # Update return normalizer
        self.return_normalizer.update(lambda_returns.detach())

        # ── Critic loss (on detached latents and targets) ──
        critic_latents = latents_stack[:, :-1].reshape(-1, latents_stack.shape[-1]).detach()
        critic_targets = lambda_returns.reshape(-1).detach()
        critic_loss = self.critic.loss(critic_latents, critic_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # ── Actor loss: maximize normalized returns + entropy ──
        normalized_returns = self.return_normalizer.normalize(lambda_returns)

        entropies = torch.stack(imagined_entropies, dim=1)  # (N, H)

        # Actor maximizes expected returns through dynamics gradients
        actor_loss = -normalized_returns.mean()
        entropy_loss = -self.entropy_coef * entropies.mean()
        total_actor_loss = actor_loss + entropy_loss

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        return {
            'actor': actor_loss.item(),
            'critic': critic_loss.item(),
            'entropy': entropies.mean().item(),
        }

    def _compute_lambda_returns(self, rewards, values, continues):
        """Compute GAE-style lambda-returns for imagined trajectories.

        lambda-return combines multi-step returns at different horizons:
        G_t^λ = r_t + γ*c_t * ((1-λ)*V_{t+1} + λ*G_{t+1}^λ)

        where c_t is the continuation probability (1 - done).
        This balances bias (short horizon, low λ) vs variance (long horizon, high λ).
        """
        H = rewards.shape[1]
        returns = torch.zeros_like(rewards)

        # Bootstrap from final value
        last_return = values[:, -1]

        for t in reversed(range(H)):
            last_return = rewards[:, t] + self.gamma * continues[:, t] * (
                (1 - self.lam) * values[:, t + 1] + self.lam * last_return
            )
            returns[:, t] = last_return

        return returns

    def _update_target(self, tau=0.02):
        """Soft-update critic target network."""
        for param, target_param in zip(self.critic.parameters(),
                                        self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    @torch.no_grad()
    def predict_sequence(self, obs_seq, action_seq):
        """Predict observations and rewards for a sequence (for visualization).

        Given real observations and actions, encodes through posterior and
        then decodes — shows how well the world model can reconstruct.
        """
        self.eval()
        obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.tensor(action_seq, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.discrete and act_t.shape[-1] != self.action_dim:
            act_t = F.one_hot(act_t.long().squeeze(-1), self.action_dim).float()

        embeds = self.encoder(obs_t)
        h_init, z_init = self.rssm.initial_state(1, self.device)
        h_states, z_posts, _, _ = self.rssm.observe_sequence(embeds, act_t, h_init, z_init)

        latents = self.rssm.get_latent(h_states, z_posts)
        latents_flat = latents.reshape(-1, latents.shape[-1])

        from .utils import symexp
        pred_obs = symexp(self.decoder(latents_flat)).squeeze(0).cpu().numpy()
        pred_rewards = twohot_decode(
            self.reward_head(latents_flat), self.num_bins
        ).squeeze(0).cpu().numpy()

        self.train()
        return pred_obs, pred_rewards
