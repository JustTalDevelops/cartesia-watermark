"""Experimental CPU packing for the Hindi/Tamil Large-V3 primary recognizer.

The active frozen study uses the original float32 recognizer. Packed weights
need a new original cache identity and their own prospective validation.
"""
from pathlib import Path

from codec_audio import digest
from codec_content import ContentRecognizer


def pack_linear_weights(model):
    import torch
    if torch.backends.quantized.engine not in ('x86', 'fbgemm'):
        raise RuntimeError('Experimental packed FP16 requires an x86/fbgemm CPU backend')
    names = {name for name, module in model.named_modules()
             if type(module) is torch.nn.Linear and name != 'proj_out'}
    if not names:
        raise ValueError('No supported linear layers to pack')
    packed = torch.ao.quantization.quantize_dynamic(model, names, dtype=torch.float16, inplace=False).eval()
    count = sum(isinstance(module, torch.ao.nn.quantized.dynamic.Linear) for module in packed.modules())
    if count != len(names):
        raise RuntimeError('Some requested linear layers were not packed')
    return packed, count


class PackedLargeRecognizer(ContentRecognizer):
    def __init__(self):
        import torch
        if torch.backends.quantized.engine not in ('x86', 'fbgemm'):
            raise RuntimeError('Experimental packed FP16 requires an x86/fbgemm CPU backend')
        super().__init__()
        self._packed_large_reference = None
        # Keep metadata fixed before any source is recognized: original caches
        # can be created before the first Hindi/Tamil source loads Large V3.
        self.metadata['guard_version'] = 'prospective-v3-large-packed-fp16-experimental'
        self.metadata['precision'] = ('Large linear weights packed FP16 with float32 inputs/outputs; '
                                      'Turbo/MMS/SenseVoice float32; pinned FastConformer INT8')
        self.metadata['large'] = dict(self.metadata['large'], weight_packing={
            'weights': 'torch.float16', 'inputs_outputs': 'torch.float32',
            'policy': 'All torch.nn.Linear except proj_out', 'linear_modules': 512,
            'engine': torch.backends.quantized.engine,
            'implementation_sha256': digest(Path(__file__))})

    def _large(self):
        processor, model = super()._large()
        if model is not self._packed_large_reference:
            packed, count = pack_linear_weights(model)
            if count != self.metadata['large']['weight_packing']['linear_modules']:
                raise RuntimeError('Packed model differs from the pinned Large-V3 architecture')
            self.large_model = self._packed_large_reference = packed
        return processor, self.large_model
