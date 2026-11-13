"""Regression checks for adaptive feedback; no audio files or external calls."""
import unittest

import torch

from remover import Remover
from train_adaptive_remover import Critic, discriminator_loss


class AdaptiveFeedbackTest(unittest.TestCase):
    def test_constant_answer_has_no_class_prior_gradient(self):
        for offset in (-.8, 0., .8):
            bias=torch.tensor(offset,requires_grad=True)
            loss=discriminator_loss(bias.expand(2),bias.expand(2),bias.expand(2))
            loss.backward()
            self.assertAlmostEqual(float(loss.detach()),2.,places=6)
            self.assertAlmostEqual(float(bias.grad),0.,places=6)

    def test_frozen_critic_preserves_input_gradient(self):
        torch.set_num_threads(2);torch.manual_seed(2081)
        critic=Critic(Remover().bandpass_fir).eval().requires_grad_(False)
        audio=torch.randn(2,16000,requires_grad=True)
        score=critic(audio)
        self.assertEqual(tuple(score.shape),(2,))
        score.sum().backward()
        self.assertTrue(bool(torch.isfinite(audio.grad).all()))
        self.assertGreater(float(audio.grad.abs().sum()),0.)


if __name__=='__main__':
    unittest.main()
