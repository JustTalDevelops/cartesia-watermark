"""Check that prior filtering does not replace the original speech condition."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from reconstruct_filtered_prior import FilteredPrior


class ConditionSpy(torch.nn.Module):
    def forward(self, condition, target=None, **kwargs):
        self.condition, self.prior = condition.clone(), kwargs['ψ0'].clone()
        if getattr(self, 'fail', False):
            raise ValueError('Simulated model failure')
        return condition


class IdentityNormalizer(torch.nn.Module):
    def forward(self, x, update=False):
        return x


class FakeEnhancer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hp = SimpleNamespace(wav_rate=16000)
        self.lcfm, self.normalizer = ConditionSpy(), IdentityNormalizer()

    def to_mel(self, x):
        return x[:, None, :]

    def forward(self, x):
        self.received = x.clone()
        mel = self.to_mel(x/(x.abs().max(dim=-1, keepdim=True).values+1e-7))
        return self.lcfm(mel, None, **{'ψ0': mel})[:, 0, :]


class FilteredPriorTests(unittest.TestCase):
    def setUp(self):
        self.base = FakeEnhancer()
        self.model = FilteredPrior(self.base).eval()
        t = torch.arange(16000)/16000
        self.x = (.3*torch.sin(2*torch.pi*1000*t)+.3*torch.sin(2*torch.pi*6000*t))[None]

    def test_original_condition_is_exact_and_only_prior_loses_upper_band(self):
        output = self.model(self.x)
        expected = self.x/(self.x.abs().max()+1e-7)
        self.assertTrue(torch.equal(self.base.received, self.x))
        self.assertTrue(torch.equal(output, expected))
        self.assertTrue(torch.equal(self.base.lcfm.condition[:, 0], expected))
        before = expected[0].numpy()[3200:12800]
        after = self.base.lcfm.prior[0, 0].numpy()[3200:12800]
        time = np.arange(3200, 12800)/16000
        for frequency, minimum, maximum in ((1000, .99, 1.01), (6000, 0., .001)):
            basis = np.sin(2*np.pi*frequency*time)
            ratio = abs(after@basis/(before@basis))
            self.assertGreaterEqual(ratio, minimum)
            self.assertLess(ratio, maximum)
        self.assertIsNone(self.model._prior)

    def test_failed_call_clears_prior_before_next_recording(self):
        self.base.lcfm.fail = True
        with self.assertRaises(ValueError):
            self.model(self.x)
        self.assertIsNone(self.model._prior)
        self.base.lcfm.fail = False
        self.assertEqual(self.model(self.x).shape, self.x.shape)


if __name__ == '__main__':
    unittest.main()
