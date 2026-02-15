"""
DreamerV3 Neural Network Components
====================================
All prediction heads that sit on top of the RSSM latent state:
- Encoder: obs → embedding (for posterior computation)
- Decoder: latent → obs reconstruction (symlog MSE)
- RewardHead: latent → reward distribution (twohot)
- ContinueHead: latent → episode continuation probability (Bernoulli)
- Actor: latent → action distribution (Categorical or squashed Normal)
- Critic: latent → value distribution (twohot)

Each head takes the concatenated RSSM state [h_t; z_t.flatten()] as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math

from .utils import symlog, twohot_encode


class MLP(nn.Module):
    """Simple feedforward network with LayerNorm + SiLU activation.

    DreamerV3 uses LayerNorm (not BatchNorm) for stability in RL,
    and SiLU (swish) as the activation function.
    """

    def __init__(self, in_dim, out_dim, hidden=256, layers=2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers + [out_dim]
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on output layer
                modules.append(nn.LayerNorm(dims[i+1]))
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """Observation encoder: obs → embedding vector.

    For low-dimensional observations (CartPole, Pendulum), a simple MLP.
    For pixel observations, this would be a CNN — left as MLP for clarity.
    """

    def __init__(self, obs_dim, embed_dim=256, hidden=256):
        super().__init__()
        self.net = MLP(obs_dim, embed_dim, hidden=hidden, layers=2)

    def forward(self, obs):
        return self.net(obs)


class Decoder(nn.Module):
    """Observation decoder: latent → reconstructed observation.

    Predicts observations in symlog space — the loss is MSE between
    symlog(real_obs) and decoder output, which naturally handles
    observations of different magnitudes.
    """

    def __init__(self, latent_dim, obs_dim, hidden=256):
        super().__init__()
        self.net = MLP(latent_dim, obs_dim, hidden=hidden, layers=2)

    def forward(self, latent):
        """Returns predicted observation in symlog space."""
        return self.net(latent)

    def loss(self, latent, target_obs):
        """Symlog MSE loss: ||decoder(latent) - symlog(target)||^2."""
        pred = self.forward(latent)
        target = symlog(target_obs)
        return F.mse_loss(pred, target)


class RewardHead(nn.Module):
    """Reward predictor using twohot discrete regression.

    Instead of predicting a single scalar (fragile, sparse gradients),
    DreamerV3 predicts a distribution over 255 bins. This gives richer
    learning signal — every bin gets a gradient, not just one number.
    """

    def __init__(self, latent_dim, num_bins=255, hidden=256):
        super().__init__()
        self.num_bins = num_bins
        self.net = MLP(latent_dim, num_bins, hidden=hidden, layers=2)

    def forward(self, latent):
        """Returns logits over reward bins."""
        return self.net(latent)

    def loss(self, latent, target_reward):
        """Cross-entropy between predicted bin logits and twohot target."""
        logits = self.forward(latent)
        target = twohot_encode(target_reward, self.num_bins)
        # Cross-entropy with soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(-1).mean()


class ContinueHead(nn.Module):
    """Episode continuation predictor (1 - done probability).

    Predicts whether the episode continues at the next step.
    Used during imagination to know when to stop trajectories.
    """

    def __init__(self, latent_dim, hidden=256):
        super().__init__()
        self.net = MLP(latent_dim, 1, hidden=hidden, layers=2)

    def forward(self, latent):
        return self.net(latent).squeeze(-1)

    def loss(self, latent, target_continue):
        """Weighted BCE loss: predict P(episode continues).

        Terminal transitions are rare (~3-5% of data), so we weight
        them 10x higher to ensure the model learns when episodes end.
        Without this, the model learns to always predict "continue"
        and the actor gets no signal about terminal states.
        """
        logits = self.forward(latent)
        weight = torch.where(target_continue < 0.5, 10.0, 1.0)
        return F.binary_cross_entropy_with_logits(logits, target_continue, weight=weight)


class Actor(nn.Module):
    """Policy network: latent → action distribution.

    Discrete actions: outputs Categorical distribution logits.
    Continuous actions: outputs mean and log_std for squashed Normal.

    Entropy regularization encourages exploration — DreamerV3 targets
    a specific entropy level rather than using a fixed coefficient.
    """

    def __init__(self, latent_dim, action_dim, discrete=True, hidden=256):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim

        if discrete:
            self.net = MLP(latent_dim, action_dim, hidden=hidden, layers=2)
        else:
            self.net = MLP(latent_dim, action_dim * 2, hidden=hidden, layers=2)

    def forward(self, latent):
        """Returns action distribution."""
        if self.discrete:
            logits = self.net(latent)
            return D.Categorical(logits=logits)
        else:
            out = self.net(latent)
            mean, log_std = out.chunk(2, dim=-1)
            log_std = log_std.clamp(-5, 2)
            std = log_std.exp()
            return D.Normal(mean, std)

    def sample(self, latent):
        """Sample action with straight-through gradients (discrete) or reparameterized (continuous)."""
        dist = self.forward(latent)
        if self.discrete:
            # Straight-through: sample but pass gradient through logits
            sample = dist.sample()
            # One-hot with straight-through gradient
            one_hot = F.one_hot(sample, self.action_dim).float()
            # Straight-through: forward pass uses one_hot, backward uses probs
            action = one_hot + dist.probs - dist.probs.detach()
            return sample, action, dist
        else:
            # Reparameterized sample
            raw = dist.rsample()
            action = torch.tanh(raw)  # Squash to [-1, 1]
            return action, action, dist

    def log_prob(self, latent, action):
        """Compute log probability of action under current policy."""
        dist = self.forward(latent)
        if self.discrete:
            return dist.log_prob(action)
        else:
            # Log prob with tanh squashing correction
            raw = torch.atanh(action.clamp(-0.999, 0.999))
            log_prob = dist.log_prob(raw).sum(-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
            return log_prob

    def entropy(self, latent):
        """Compute entropy of action distribution."""
        dist = self.forward(latent)
        if self.discrete:
            return dist.entropy()
        else:
            return dist.entropy().sum(-1)


class Critic(nn.Module):
    """Value predictor using twohot discrete regression.

    Same twohot approach as RewardHead — predicts a distribution
    over value bins rather than a single scalar. Targets are
    lambda-returns transformed through symlog.
    """

    def __init__(self, latent_dim, num_bins=255, hidden=256):
        super().__init__()
        self.num_bins = num_bins
        self.net = MLP(latent_dim, num_bins, hidden=hidden, layers=2)

    def forward(self, latent):
        """Returns logits over value bins."""
        return self.net(latent)

    def loss(self, latent, target_value):
        """Cross-entropy loss with twohot-encoded target values."""
        logits = self.forward(latent)
        target = twohot_encode(target_value, self.num_bins)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(-1).mean()
