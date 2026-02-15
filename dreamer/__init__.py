"""
DreamerV3 — World Model Reinforcement Learning from Scratch
============================================================
Educational implementation of DreamerV3 (Hafner et al., Nature 2025).

Core idea: Learn a world model from experience, then train the agent
entirely "in imagination" — no environment interaction needed for
policy improvement.

Components:
- RSSM: Recurrent State-Space Model (learned dynamics)
- Actor-Critic: Trained on imagined trajectories
- Symlog + Twohot: Scale-invariant regression techniques
"""

from .utils import symlog, symexp, twohot_encode, twohot_decode
from .utils import get_device, set_seed, ReturnNormalizer
from .utils import plot_training_curves, plot_world_model_predictions
from .networks import MLP, Encoder, Decoder, RewardHead, ContinueHead, Actor, Critic
from .rssm import RSSM
from .replay_buffer import SequenceReplayBuffer
from .agent import DreamerAgent

__all__ = [
    'symlog', 'symexp', 'twohot_encode', 'twohot_decode',
    'get_device', 'set_seed', 'ReturnNormalizer',
    'plot_training_curves', 'plot_world_model_predictions',
    'MLP', 'Encoder', 'Decoder', 'RewardHead', 'ContinueHead',
    'Actor', 'Critic', 'RSSM', 'SequenceReplayBuffer', 'DreamerAgent',
]
