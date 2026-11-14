import types
import unittest

import torch

from whisper_memory import install_encoder_attention_reduction


class Attention(torch.nn.Module):
    def forward(self, hidden, attention, cache):
        return hidden, attention, cache


def model(layers):
    return types.SimpleNamespace(config=types.SimpleNamespace(model_type='whisper'),
        model=types.SimpleNamespace(encoder=types.SimpleNamespace(layers=layers)))


class WhisperMemoryTests(unittest.TestCase):
    def test_releases_diagnostic_storage_and_preserves_model_outputs(self):
        attention = Attention()
        handles = install_encoder_attention_reduction(model([types.SimpleNamespace(self_attn=attention)]))
        hidden, weights, cache = torch.randn(2, 7, 5), torch.randn(2, 4, 7, 7), object()
        actual = attention(hidden, weights, cache)
        self.assertIs(actual[0], hidden)
        self.assertIs(actual[2], cache)
        self.assertEqual(actual[1].shape, (2, 0, 0, 0))
        self.assertEqual(actual[1].untyped_storage().nbytes(), 0)
        self.assertIsNone(attention(hidden, None, cache)[1])
        handles[0].remove()
        self.assertIs(attention(hidden, weights, cache)[1], weights)

    def test_rolls_back_partial_installation(self):
        attention = Attention()
        with self.assertRaises(AttributeError):
            install_encoder_attention_reduction(model([types.SimpleNamespace(self_attn=attention), object()]))
        self.assertEqual(len(attention._forward_hooks), 0)
        bad = model([])
        with self.assertRaises(ValueError):
            install_encoder_attention_reduction(bad)
        bad.config.model_type = 'other'
        with self.assertRaises(ValueError):
            install_encoder_attention_reduction(bad)


if __name__ == '__main__':
    unittest.main()
