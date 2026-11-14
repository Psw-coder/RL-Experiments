"""PPO clipped surrogate and GAE lambda-return critic targets."""
import torch
from rpo import targets as rpo_targets


def targets(rewards, values, next_values, terminated, truncated, gamma, lam):
    advantages, _ = rpo_targets(
        rewards, values, next_values, terminated, truncated, gamma, lam)
    return advantages, advantages + values


def policy_loss(logp, old_logp, advantage, clip=0.1):
    ratio = (logp - old_logp).exp()
    return -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
