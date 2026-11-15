"""Ensure the optional prefix context restores a worker after failures."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.xcodec2_prefix import semantic_prefix


class Layer:
    def __init__(self): self.hooks = []
    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))


def model():
    layers = [Layer() for _ in range(24)]
    def original(value):
        for hook in list(layers[16].hooks): hook(layers[16], (value,))
        return 'full result'
    semantic = SimpleNamespace(training=False, encoder=SimpleNamespace(layers=layers),
        config=SimpleNamespace(output_hidden_states=True), forward=original, adapter=None)
    return SimpleNamespace(semantic_model=semantic), original


class PrefixTests(unittest.TestCase):
    def test_exact_boundary_object_and_remaining_rng_draws(self):
        codec, original = model(); draws = []; value = object()
        with patch.dict(sys.modules, torch=SimpleNamespace(rand=lambda shape: draws.append(shape))):
            with semantic_prefix(codec):
                result = codec.semantic_model.forward(value)
                self.assertIs(result.hidden_states[16], value)
            self.assertEqual(draws, [[] for _ in range(7)])
        self.assertIs(codec.semantic_model.forward, original)
        self.assertEqual(codec.semantic_model.encoder.layers[16].hooks, [])

    def test_body_exception_restores_original_forward(self):
        codec, original = model()
        with self.assertRaisesRegex(RuntimeError, 'body'):
            with semantic_prefix(codec): raise RuntimeError('body')
        self.assertIs(codec.semantic_model.forward, original)

    def test_author_exception_restores_hook_and_forward(self):
        codec, _ = model()
        def failing(value): raise LookupError('author failed')
        codec.semantic_model.forward = failing
        with self.assertRaises(LookupError):
            with semantic_prefix(codec): codec.semantic_model.forward(object())
        self.assertIs(codec.semantic_model.forward, failing)
        self.assertEqual(codec.semantic_model.encoder.layers[16].hooks, [])

    def test_unreached_boundary_is_rejected(self):
        codec, _ = model(); codec.semantic_model.forward = lambda value: 'no hook'
        with self.assertRaises(RuntimeError):
            with semantic_prefix(codec): codec.semantic_model.forward(object())
        self.assertEqual(codec.semantic_model.encoder.layers[16].hooks, [])

    def test_wrong_model_configuration_is_rejected(self):
        codec, original = model(); codec.semantic_model.training = True
        with self.assertRaises(ValueError):
            with semantic_prefix(codec): pass
        self.assertIs(codec.semantic_model.forward, original)


if __name__ == '__main__': unittest.main()
