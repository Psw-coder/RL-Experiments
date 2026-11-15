"""Reflective Policy Optimization, arXiv:2406.03678v1, equations (5)-(7)."""
import runtime
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class Agent(nn.Module):
    def __init__(self, actions=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU())
        self.actor = nn.Linear(512, actions)
        self.critic = nn.Linear(512, 1)
        # Match the authors' DeepRL layer_init (rpo_atari/deep_rl/network):
        # gain 1.0 for the encoder, 1e-3 for both heads.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, 1.0)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 1e-3)
        nn.init.orthogonal_(self.critic.weight, 1e-3)

    def forward(self, obs):
        features = self.encoder(obs.float() / 255.0)
        return Categorical(logits=self.actor(features)), self.critic(features).squeeze(-1)


def targets(rewards, values, next_values, terminated, truncated, gamma, lam):
    """GAE and discounted critic returns (as in the author's Atari implementation).

    True termination disables bootstrap; time limits bootstrap terminal observations
    but stop recursion so reset episodes never leak into the old episode.
    """
    advantage = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    ret = next_values[-1].copy()
    for t in reversed(range(len(rewards))):
        live = 1.0 - terminated[t]
        continuation = 1.0 - np.logical_or(terminated[t], truncated[t])
        delta = rewards[t] + gamma * live * next_values[t] - values[t]
        carry = delta + gamma * lam * continuation * carry
        advantage[t] = carry
        ret = rewards[t] + gamma * live * np.where(truncated[t], next_values[t], ret)
        returns[t] = ret
    return advantage, returns


def pair_indices(indices, num_envs, rollout_length, done):
    """Time-major layout: the next time step of SAME worker is index + N."""
    following = indices + num_envs
    valid = (following < num_envs * rollout_length) & ~done[indices]
    return np.minimum(following, num_envs * rollout_length - 1), valid


def policy_loss(logp, old_logp, advantage, next_logp, next_old_logp,
                next_advantage, valid, beta=3.0, clip=0.1, next_clip=0.1):
    ratio = (logp - old_logp).exp()
    ratio_next = (next_logp - next_old_logp).exp()
    clipped = ratio.clamp(1 - clip, 1 + clip)
    clipped_next = ratio_next.clamp(1 - next_clip, 1 + next_clip)
    ppo = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    reflective = -(torch.minimum(ratio * ratio_next * next_advantage,
                        clipped * clipped_next * next_advantage) * valid).mean()
    return ppo + beta * reflective, ppo, reflective
