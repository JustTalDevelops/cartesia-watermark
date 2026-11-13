"""Waveform surrogate trained on real endpoint labels, not model provenance.

AudioSeal provides the pretrained convolutional encoder. Its original watermark
classes are replaced and the entire network is adapted to Cartesia labels.
This is a research teacher, not a claim of compatibility between watermarks.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RawOracle(nn.Module):
    def __init__(self):
        super().__init__()
        from audioseal import AudioSeal
        detector = AudioSeal.load_detector('audioseal_detector_16bits').detector
        channels = detector[-1].in_channels
        detector[-1] = nn.Conv1d(channels, 2, 1)
        nn.init.normal_(detector[-1].weight, std=.01)
        nn.init.zeros_(detector[-1].bias)
        self.detector = detector

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = torch.full((len(x),), x.shape[-1], device=x.device)
        valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
        power = (x.square()*valid).sum(-1)/lengths.clamp_min(1)
        z = x * (.05 / power.sqrt().clamp_min(1e-5))[:, None]
        logits = self.detector(z[:, None])
        difference = logits[:, 1]-logits[:, 0]
        # Pool 20 ms frames, suppress padded batch tails, then use a smooth
        # maximum so partial watermark segments can retain a positive label.
        scores = F.avg_pool1d(difference[:, None], 320, 320)[:, 0]
        frames = (lengths//320).clamp_min(1)
        mask = torch.arange(scores.shape[-1], device=x.device)[None, :] < frames[:, None]
        return .5*(torch.logsumexp((scores/.5).masked_fill(~mask, -1e4), -1)-frames.log())


def load_raw_oracle(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'cartesia-raw-oracle-v1':
        raise ValueError('Unsupported oracle checkpoint')
    model = RawOracle()
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint
