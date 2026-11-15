"""Experimental exact-prefix execution for X-Codec2's hidden-state-16 consumer."""
from contextlib import contextmanager
from types import SimpleNamespace


class PrefixComplete(Exception):
    pass


@contextmanager
def semantic_prefix(model):
    import torch
    semantic = model.semantic_model
    layers = semantic.encoder.layers
    if semantic.training or len(layers) != 24 or semantic.config.output_hidden_states is not True:
        raise ValueError('Only the pinned eval-mode 24-layer semantic model is supported')
    if getattr(semantic, 'adapter', None) is not None:
        raise ValueError('A downstream semantic adapter requires a separate proof')
    original = semantic.forward
    record = {}
    def stop(module, arguments):
        raise PrefixComplete(arguments[0])
    def forward(*args, **kwargs):
        if len(args) != 1 or kwargs:
            raise ValueError('Only the pinned author codec call signature is supported')
        handle = layers[16].register_forward_pre_hook(stop)
        try:
            original(*args)
        except PrefixComplete as result:
            value = result.args[0]
            # The encoder draws LayerDrop randomness even in eval mode. Keep
            # the original RNG advancement as well as its used hidden state.
            for _ in range(7): torch.rand([])
            record['hidden_state_16'] = value
            return SimpleNamespace(hidden_states=(None,)*16+(value,))
        else:
            raise RuntimeError('Expected semantic layer boundary was not reached')
        finally:
            handle.remove()
    semantic.forward = forward
    try:
        yield record
    finally:
        semantic.forward = original
