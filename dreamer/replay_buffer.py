"""
Sequence Replay Buffer
=======================
DreamerV3 needs sequences (not individual transitions) for training
the RSSM world model. This buffer stores complete episodes and samples
fixed-length subsequences for batch training.

Key difference from standard replay buffers:
- Stores episodes, not individual (s, a, r, s') tuples
- Samples contiguous subsequences of length T from random episodes
- Returns a validity mask so padded steps don't corrupt training
- Maintains temporal structure needed for recurrent world model
"""

import numpy as np
import torch


class SequenceReplayBuffer:
    """Episode-based replay buffer that samples fixed-length sequences."""

    def __init__(self, max_episodes=1000):
        self.max_episodes = max_episodes
        self.episodes = []
        self.current_episode = {
            'obs': [], 'actions': [], 'rewards': [], 'dones': []
        }

    def add(self, obs, action, reward, done):
        """Add a single transition to the current episode."""
        self.current_episode['obs'].append(obs)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['dones'].append(done)

        if done:
            self._store_episode()

    def _store_episode(self):
        """Convert current episode to numpy arrays and store."""
        if len(self.current_episode['obs']) < 2:
            self.current_episode = {
                'obs': [], 'actions': [], 'rewards': [], 'dones': []
            }
            return

        episode = {
            'obs': np.array(self.current_episode['obs'], dtype=np.float32),
            'actions': np.array(self.current_episode['actions'], dtype=np.float32),
            'rewards': np.array(self.current_episode['rewards'], dtype=np.float32),
            'dones': np.array(self.current_episode['dones'], dtype=np.float32),
        }

        self.episodes.append(episode)

        # Remove oldest episodes if over capacity
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

        self.current_episode = {
            'obs': [], 'actions': [], 'rewards': [], 'dones': []
        }

    def sample(self, batch_size, seq_len, device='cpu'):
        """Sample a batch of sequences from stored episodes.

        Returns dict with 'obs', 'actions', 'rewards', 'dones', 'mask'.
        The 'mask' tensor is 1.0 for valid (real) steps and 0.0 for
        padded steps, preventing padding from corrupting training.
        """
        batch = {'obs': [], 'actions': [], 'rewards': [], 'dones': [], 'mask': []}

        for _ in range(batch_size):
            ep = self.episodes[np.random.randint(len(self.episodes))]
            ep_len = len(ep['obs'])
            max_start = max(0, ep_len - seq_len)
            start = np.random.randint(0, max_start + 1)
            end = start + seq_len
            actual_len = min(seq_len, ep_len - start)

            for key in ['obs', 'actions', 'rewards', 'dones']:
                chunk = ep[key][start:end]
                if len(chunk) < seq_len:
                    pad_shape = (seq_len - len(chunk),) + chunk.shape[1:]
                    chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=np.float32)])
                batch[key].append(chunk)

            # Validity mask: 1.0 for real data, 0.0 for padding
            mask = np.zeros(seq_len, dtype=np.float32)
            mask[:actual_len] = 1.0
            batch['mask'].append(mask)

        return {k: torch.tensor(np.array(v), device=device)
                for k, v in batch.items()}

    @property
    def total_steps(self):
        """Total number of transitions stored."""
        return sum(len(ep['obs']) for ep in self.episodes)

    def __len__(self):
        return len(self.episodes)
