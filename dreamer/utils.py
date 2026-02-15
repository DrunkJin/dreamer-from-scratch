"""
DreamerV3 Utility Functions
===========================
Core mathematical tools that make DreamerV3 work across diverse environments:
- Symlog/Symexp: symmetric log transform for handling varied reward scales
- Twohot encoding: soft discretization for robust value regression
- Percentile return normalization: stabilizes actor-critic training
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import os


# ─── Device ────────────────────────────────────────────────────────────

def get_device():
    """Get best available device: CUDA > CPU.

    Note: MPS (Apple Silicon) is disabled by default due to gradient
    computation issues in PyTorch < 2.3. Set DREAMER_USE_MPS=1 to enable.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (os.environ.get("DREAMER_USE_MPS") == "1"
            and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Symlog Transform ──────────────────────────────────────────────────

def symlog(x):
    """Symmetric logarithmic transform: sign(x) * ln(|x| + 1).

    Compresses large values while preserving sign — crucial for DreamerV3
    to handle environments with vastly different reward scales (e.g., Atari
    scores of 0-100k vs continuous control rewards of -10 to 0).

    Uses torch.where instead of sign()*abs() for MPS compatibility and
    clean gradient flow.
    """
    return torch.where(x >= 0, torch.log1p(x), -torch.log1p(-x))


def symexp(x):
    """Inverse of symlog: sign(x) * (exp(|x|) - 1).

    Uses torch.where for MPS compatibility.
    """
    return torch.where(x >= 0, torch.exp(x) - 1, 1 - torch.exp(-x))


# ─── Twohot Encoding ───────────────────────────────────────────────────

def twohot_encode(x, num_bins=255, low=-20.0, high=20.0):
    """Encode scalar values as soft one-hot vectors over discrete bins.

    Instead of regressing a single scalar (which gives sparse gradients),
    DreamerV3 predicts a distribution over bins. Each value activates the
    two nearest bins proportionally — like a soft histogram.

    Args:
        x: Scalar values in ORIGINAL space (before symlog)
        num_bins: Number of discrete bins (255 in paper)
        low, high: Range in original space

    Returns:
        Twohot vector of shape (*x.shape, num_bins)
    """
    # Transform to symlog space for the bin boundaries
    symlog_low = symlog(torch.tensor(low, dtype=torch.float32)).item()
    symlog_high = symlog(torch.tensor(high, dtype=torch.float32)).item()
    bins = torch.linspace(symlog_low, symlog_high, num_bins, device=x.device)

    # Transform input to symlog space
    x_symlog = symlog(x)

    # Clamp to bin range
    x_symlog = x_symlog.clamp(symlog_low, symlog_high)

    # Find which two bins the value falls between
    below = (bins.unsqueeze(0) <= x_symlog.unsqueeze(-1)).sum(-1) - 1
    below = below.clamp(0, num_bins - 2)
    above = below + 1

    # Compute interpolation weight
    below_val = bins[below]
    above_val = bins[above]
    weight = (x_symlog - below_val) / (above_val - below_val + 1e-8)
    weight = weight.clamp(0, 1)

    # Build twohot vector
    twohot = torch.zeros(*x.shape, num_bins, device=x.device)
    twohot.scatter_(-1, below.unsqueeze(-1), (1 - weight).unsqueeze(-1))
    twohot.scatter_(-1, above.unsqueeze(-1), weight.unsqueeze(-1))

    return twohot


def twohot_decode(logits, num_bins=255, low=-20.0, high=20.0):
    """Decode twohot logits back to scalar values.

    Takes softmax over bin logits, then computes expected value
    as the probability-weighted sum of bin centers (in symlog space),
    finally transformed back via symexp.
    """
    symlog_low = symlog(torch.tensor(low, dtype=torch.float32)).item()
    symlog_high = symlog(torch.tensor(high, dtype=torch.float32)).item()
    bins = torch.linspace(symlog_low, symlog_high, num_bins, device=logits.device)

    probs = F.softmax(logits, dim=-1)
    symlog_value = (probs * bins).sum(-1)
    return symexp(symlog_value)


# ─── Return Normalization ──────────────────────────────────────────────

class ReturnNormalizer:
    """Percentile-based return normalization (DreamerV3 Section 3).

    Instead of dividing by running std (which fails with sparse rewards),
    DreamerV3 normalizes by the range between the 5th and 95th percentile
    of recent returns: scale = max(1, P95 - P5).

    The max(1, ...) ensures we never amplify small-scale returns.
    """

    def __init__(self, decay=0.99, percentile_low=5, percentile_high=95):
        self.decay = decay
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.low = None
        self.high = None

    def update(self, returns):
        """Update running percentile estimates with EMA."""
        low = torch.quantile(returns.detach().float(), self.percentile_low / 100)
        high = torch.quantile(returns.detach().float(), self.percentile_high / 100)

        if self.low is None:
            self.low = low.item()
            self.high = high.item()
        else:
            self.low = self.decay * self.low + (1 - self.decay) * low.item()
            self.high = self.decay * self.high + (1 - self.decay) * high.item()

    def normalize(self, returns):
        """Normalize returns by percentile range, floored at 1."""
        if self.low is None:
            return returns
        scale = max(1.0, self.high - self.low)
        return returns / scale


# ─── Plotting Helpers ───────────────────────────────────────────────────

def plot_training_curves(rewards, losses, title, save_path):
    """Plot reward curve and loss curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward curve with moving average
    axes[0].plot(rewards, alpha=0.3, color='steelblue', label='Episode Return')
    if len(rewards) >= 10:
        window = min(20, len(rewards) // 3)
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), ma, color='darkblue',
                     linewidth=2, label=f'{window}-ep Moving Avg')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title(f'{title} — Episode Returns')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss curves
    for name, vals in losses.items():
        axes[1].plot(vals, label=name, alpha=0.8)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{title} — Losses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('symlog')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_world_model_predictions(real_obs, pred_obs, real_rewards, pred_rewards,
                                  title, save_path):
    """Compare world model predictions against real observations."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    T = min(len(real_obs), len(pred_obs))

    # Observation prediction (first few dims)
    ndim = min(4, real_obs.shape[-1]) if real_obs.ndim > 1 else 1
    for d in range(ndim):
        r = real_obs[:T, d] if ndim > 1 else real_obs[:T]
        p = pred_obs[:T, d] if ndim > 1 else pred_obs[:T]
        axes[0].plot(r, label=f'Real dim {d}', alpha=0.7)
        axes[0].plot(p, '--', label=f'Pred dim {d}', alpha=0.7)
    axes[0].set_title(f'{title} — Observation Prediction')
    axes[0].set_xlabel('Time Step')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Reward prediction
    T_r = min(len(real_rewards), len(pred_rewards))
    axes[1].plot(real_rewards[:T_r], label='Real Reward', alpha=0.7, color='steelblue')
    axes[1].plot(pred_rewards[:T_r], '--', label='Predicted Reward', alpha=0.7, color='coral')
    axes[1].set_title(f'{title} — Reward Prediction')
    axes[1].set_xlabel('Time Step')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
