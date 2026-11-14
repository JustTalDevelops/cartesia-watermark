"""Reduce unused encoder attention diagnostics in the private Whisper guard.

Word alignment uses decoder cross attention. Transformers also retains every
encoder self-attention matrix when that diagnostic flag is enabled, including
for every long-form segment. These matrices do not contribute to generation or
word alignment after their attention layer has computed its output.

Do not use this helper when callers need encoder attention visualizations.
"""


def install_encoder_attention_reduction(model):
    if getattr(model.config, 'model_type', None) != 'whisper':
        raise ValueError('Encoder attention reduction requires a Whisper model')
    layers = model.model.encoder.layers
    if not layers:
        raise ValueError('Whisper encoder has no layers')

    def discard_unused_attention(module, args, output):
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError('Unexpected pinned Whisper attention output')
        hidden, attention, cache = output
        if attention is None:
            return output
        if attention.ndim != 4:
            raise RuntimeError('Unexpected pinned Whisper encoder attention shape')
        # An empty allocation, rather than a view, releases the original
        # storage. Preserve the batch axis for Transformers' long-form split.
        empty = attention.new_empty((attention.shape[0], 0, 0, 0))
        return hidden, empty, cache

    handles = []
    try:
        for layer in layers:
            handles.append(layer.self_attn.register_forward_hook(discard_unused_attention))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles
