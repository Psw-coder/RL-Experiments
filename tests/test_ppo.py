import unittest

import numpy as np
import torch

from ppo import policy_loss, targets
from rpo import pair_indices, policy_loss as reflective_policy_loss
from rpo import targets as discounted_targets


class PPOTests(unittest.TestCase):
    def test_zero_beta_matches_ppo_loss_and_gradients(self):
        logp = torch.tensor([1.5, .5, 1.05, .95]).log().requires_grad_()
        next_logp = torch.tensor([.8, 1.3, 1.1, .9]).log().requires_grad_()
        advantage = torch.tensor([1., -1., -1., 1.])
        old = torch.zeros(4)
        expected = policy_loss(logp, old, advantage)
        actual, _, _ = reflective_policy_loss(logp, old, advantage,
            next_logp, old, advantage.flip(0), torch.tensor([1., 0., 1., 1.]), beta=0.)
        torch.testing.assert_close(actual, expected)
        expected_grad, = torch.autograd.grad(expected, logp)
        actual_grad, next_grad = torch.autograd.grad(actual, (logp, next_logp))
        torch.testing.assert_close(actual_grad, expected_grad)
        torch.testing.assert_close(next_grad, torch.zeros_like(next_grad))

    def test_clipping_both_advantage_signs_and_gradients(self):
        logp = torch.tensor([1.5, 0.5, 1.5, 0.5]).log().requires_grad_()
        advantage = torch.tensor([1., -1., -1., 1.])
        loss = policy_loss(logp, torch.zeros(4), advantage, clip=0.2)
        self.assertAlmostEqual(loss.item(), 0.15, places=6)
        loss.backward()
        torch.testing.assert_close(logp.grad, torch.tensor([0., 0., .375, -.125]))

    def test_gae_stops_at_episode_boundaries(self):
        # Workers terminate, truncate, or continue at t=0 respectively.
        rewards = np.array([[1., 1., 1.], [100., 100., 100.]], dtype=np.float32)
        values = np.full((2, 3), 2., dtype=np.float32)
        next_values = np.full((2, 3), 3., dtype=np.float32)
        terminated = np.array([[True, False, False], [False, False, False]])
        truncated = np.array([[False, True, False], [False, False, False]])
        adv, ret = targets(rewards, values, next_values, terminated, truncated, .9, .5)
        np.testing.assert_allclose(adv[0], [-1., 1.7, 47.015], rtol=1e-6)
        np.testing.assert_allclose(adv[1], [100.7] * 3, rtol=1e-6)
        np.testing.assert_allclose(ret, adv + values)

    def test_paper_discounted_returns_bootstrap_at_truncation(self):
        # The paper/author Atari critic target: discounted returns, with a
        # final-observation bootstrap only for time-limit truncation.
        rewards = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
        values = np.zeros_like(rewards)
        next_values = np.array([[10., 20., 30.], [40., 50., 60.]], dtype=np.float32)
        terminated = np.zeros_like(rewards, dtype=bool)
        truncated = np.zeros_like(rewards, dtype=bool)
        terminated[0, 1] = True
        truncated[1, 0] = True
        adv, ret = discounted_targets(rewards, values, next_values,
                                      terminated, truncated, .9, .5)
        np.testing.assert_allclose(ret[0], [37., 2., 57.], rtol=1e-6)
        np.testing.assert_allclose(ret[1], [40., 50., 60.], rtol=1e-6)
        np.testing.assert_allclose(adv[0], [28., 2., 57.], rtol=1e-6)
        np.testing.assert_allclose(adv[1], [40., 50., 60.], rtol=1e-6)

    def test_pair_indices_use_same_worker_next_step(self):
        n, horizon = 3, 4
        indices = np.arange(n * horizon)
        done = np.zeros(n * horizon, dtype=bool)
        following, valid = pair_indices(indices, n, horizon, done)
        expected = indices + n
        np.testing.assert_array_equal(
            following, np.minimum(expected, n * horizon - 1))
        np.testing.assert_array_equal(valid, expected < n * horizon)
        done[1] = True
        _, valid = pair_indices(indices, n, horizon, done)
        self.assertFalse(valid[1])


if __name__ == '__main__':
    unittest.main()
