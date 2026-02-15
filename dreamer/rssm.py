"""
RSSM — Recurrent State-Space Model
====================================
The heart of DreamerV3's world model. RSSM maintains two types of state:

1. **Deterministic state h_t** (GRU hidden): captures temporal dependencies,
   like a memory of what happened in the past.

2. **Stochastic state z_t** (categorical): captures uncertainty about the
   current situation. Uses 16 categorical variables × 16 classes = 256 dims,
   which provides exponentially many possible states (16^16) while keeping
   gradients well-behaved.

Key operations:
- observe():  h_t, z_t = f(h_{t-1}, z_{t-1}, a_{t-1}, obs_t)  — uses real observation
- imagine():  h_t, ẑ_t = f(h_{t-1}, z_{t-1}, a_{t-1})         — no observation (dreaming!)

The posterior (observe) uses encoder(obs) to ground predictions in reality.
The prior (imagine) relies only on the learned dynamics — this is what
enables "learning in imagination" for the actor-critic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import MLP


class RSSM(nn.Module):
    """Recurrent State-Space Model with categorical latent variables."""

    def __init__(self, obs_dim, action_dim, embed_dim=256,
                 gru_dim=256, stoch_dim=16, stoch_classes=16,
                 hidden=256, unimix=0.01):
        """
        Args:
            obs_dim: Observation dimensionality
            action_dim: Action dimensionality (num_actions for discrete)
            embed_dim: Encoder output dim
            gru_dim: GRU hidden state size (deterministic state h)
            stoch_dim: Number of categorical variables
            stoch_classes: Number of classes per categorical variable
            hidden: MLP hidden layer size
            unimix: Uniform mixing ratio to prevent categorical collapse
        """
        super().__init__()
        self.gru_dim = gru_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_flat = stoch_dim * stoch_classes  # 16 × 16 = 256
        self.unimix = unimix

        # Flatten z + action → GRU input
        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_flat + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # GRU: deterministic state transition
        self.gru = nn.GRUCell(hidden, gru_dim)

        # LayerNorm after GRU — critical for stability!
        # Without this, GRU hidden states can grow unbounded.
        self.gru_norm = nn.LayerNorm(gru_dim)

        # Prior: p(z_t | h_t) — predict stochastic state from deterministic only
        self.prior_net = MLP(gru_dim, self.stoch_flat, hidden=hidden, layers=1)

        # Posterior: q(z_t | h_t, embed(obs_t)) — uses real observation
        self.posterior_net = MLP(gru_dim + embed_dim, self.stoch_flat,
                                hidden=hidden, layers=1)

    @property
    def latent_dim(self):
        """Total latent state dimension = h_dim + z_flat."""
        return self.gru_dim + self.stoch_flat

    def initial_state(self, batch_size, device):
        """Zero-initialized RSSM state."""
        h = torch.zeros(batch_size, self.gru_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device)
        return h, z

    def get_latent(self, h, z):
        """Concatenate deterministic and flattened stochastic state."""
        return torch.cat([h, z.flatten(-2, -1)], dim=-1)

    def _categorical_sample(self, logits):
        """Sample from categorical with straight-through gradient and unimix.

        Unimix (1% uniform): prevents any class probability from reaching 0,
        which would kill gradients for that class permanently.

        Straight-through: forward pass uses one-hot sample, backward pass
        uses softmax probabilities — enables gradient flow through discrete choice.
        """
        # Reshape to (batch, stoch_dim, stoch_classes)
        shape = logits.shape[:-1] + (self.stoch_dim, self.stoch_classes)
        logits = logits.reshape(shape)

        # Unimix: blend softmax with uniform distribution
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / self.stoch_classes
        probs = (1 - self.unimix) * probs + self.unimix * uniform

        # Sample and straight-through gradient
        indices = torch.multinomial(probs.reshape(-1, self.stoch_classes), 1)
        one_hot = F.one_hot(indices.squeeze(-1), self.stoch_classes).float()
        one_hot = one_hot.reshape(shape)

        # Straight-through: gradient flows through probs, value comes from one_hot
        z = one_hot + probs - probs.detach()

        return z, probs

    def observe_step(self, h_prev, z_prev, action, embed):
        """Single observation step: advance state using real observation.

        1. Compute new deterministic state h_t via GRU
        2. Compute prior p(z|h) — what the model predicts without seeing obs
        3. Compute posterior q(z|h,obs) — corrected using real observation
        4. Sample z_t from posterior (used for training)

        Returns both prior and posterior for KL divergence loss.
        """
        # 1. GRU step: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        x = self.pre_gru(torch.cat([z_prev.flatten(-2, -1), action], dim=-1))
        h = self.gru(x, h_prev)
        h = self.gru_norm(h)

        # 2. Prior: p(z_t | h_t)
        prior_logits = self.prior_net(h)
        prior_z, prior_probs = self._categorical_sample(prior_logits)

        # 3. Posterior: q(z_t | h_t, embed(obs_t))
        post_logits = self.posterior_net(torch.cat([h, embed], dim=-1))
        post_z, post_probs = self._categorical_sample(post_logits)

        return h, post_z, prior_probs, post_probs

    def imagine_step(self, h_prev, z_prev, action):
        """Single imagination step: advance state WITHOUT observation.

        Used during actor-critic training — the agent "dreams" forward
        using only the learned world model, generating experience to
        train the policy without touching the real environment.
        """
        x = self.pre_gru(torch.cat([z_prev.flatten(-2, -1), action], dim=-1))
        h = self.gru(x, h_prev)
        h = self.gru_norm(h)

        prior_logits = self.prior_net(h)
        z, probs = self._categorical_sample(prior_logits)

        return h, z

    def observe_sequence(self, embeds, actions, h_init, z_init):
        """Process a full sequence of observations for world model training.

        Args:
            embeds: (B, T, embed_dim) — encoded observations
            actions: (B, T, action_dim) — actions taken
            h_init, z_init: Initial RSSM state

        Returns:
            h_states: (B, T, gru_dim)
            z_posts: (B, T, stoch_dim, stoch_classes)
            prior_probs: (B, T, stoch_dim, stoch_classes)
            post_probs: (B, T, stoch_dim, stoch_classes)
        """
        B, T = embeds.shape[:2]

        h_list, z_list = [], []
        prior_list, post_list = [], []

        h, z = h_init, z_init

        for t in range(T):
            h, z, prior_p, post_p = self.observe_step(
                h, z, actions[:, t], embeds[:, t]
            )
            h_list.append(h)
            z_list.append(z)
            prior_list.append(prior_p)
            post_list.append(post_p)

        return (
            torch.stack(h_list, dim=1),      # (B, T, gru_dim)
            torch.stack(z_list, dim=1),      # (B, T, stoch_dim, stoch_classes)
            torch.stack(prior_list, dim=1),  # (B, T, stoch_dim, stoch_classes)
            torch.stack(post_list, dim=1),   # (B, T, stoch_dim, stoch_classes)
        )

    def kl_loss(self, prior_probs, post_probs, free_bits=1.0):
        """KL divergence between posterior and prior with free bits.

        Free bits (free nats): KL = max(free_bits, KL_actual).
        This prevents KL from being pushed to zero (KL collapse),
        which would make the prior useless for imagination.

        DreamerV3 uses free_bits=1.0, meaning the model is "allowed"
        1 nat of information per categorical variable for free.
        """
        # KL per categorical variable: sum over classes
        kl = (post_probs * (torch.log(post_probs + 1e-8) -
                            torch.log(prior_probs + 1e-8))).sum(-1)

        # Free bits: max(free_bits, KL) per variable, then mean over variables
        kl = torch.clamp(kl, min=free_bits / self.stoch_dim).sum(-1)

        return kl.mean()
