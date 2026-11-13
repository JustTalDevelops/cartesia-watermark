"""24 kHz Vocos mel checkpoint adapter without a torchaudio binary dependency.

Adapted from https://github.com/gemelo-ai/vocos (MIT; see VOCOS_LICENSE).
Implements only the published mel backbone + center-padded ISTFT head. The
checkpoint's original mel filter bank and windows are loaded without changes.
"""
from pathlib import Path

import torch
from torch import nn
from huggingface_hub import hf_hub_download

MODEL = "charactr/vocos-mel-24khz"
REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"


class ConvNeXtBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwconv = nn.Conv1d(512, 512, 7, padding=3, groups=512)
        self.norm = nn.LayerNorm(512, eps=1e-6)
        self.pwconv1 = nn.Linear(512, 1536)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(1536, 512)
        self.gamma = nn.Parameter(torch.ones(512) / 8)

    def forward(self, x):
        hidden = self.norm(self.dwconv(x).transpose(1, 2))
        hidden = self.pwconv2(self.act(self.pwconv1(hidden))) * self.gamma
        return x + hidden.transpose(1, 2)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv1d(100, 512, 7, padding=3)
        self.norm = nn.LayerNorm(512, eps=1e-6)
        self.convnext = nn.ModuleList([ConvNeXtBlock() for _ in range(8)])
        self.final_layer_norm = nn.LayerNorm(512, eps=1e-6)

    def forward(self, x):
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for layer in self.convnext:
            x = layer(x)
        return self.final_layer_norm(x.transpose(1, 2))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(512, 1026)
        self.istft = nn.Module()
        self.istft.register_buffer("window", torch.hann_window(1024))

    def forward(self, x):
        magnitude, phase = self.out(x).transpose(1, 2).chunk(2, 1)
        z = magnitude.exp().clamp(max=100) * torch.complex(phase.cos(), phase.sin())
        return torch.istft(z, 1024, 256, 1024, self.istft.window, center=True)


class Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = nn.Module()
        self.mel_spec.spectrogram = nn.Module()
        self.mel_spec.spectrogram.register_buffer("window", torch.hann_window(1024))
        self.mel_spec.mel_scale = nn.Module()
        self.mel_spec.mel_scale.register_buffer("fb", torch.zeros(513, 100))

    def forward(self, x):
        z = torch.stft(x, 1024, 256, window=self.mel_spec.spectrogram.window,
                       return_complex=True, center=True, pad_mode="reflect")
        mel = (z.abs().transpose(1, 2) @ self.mel_spec.mel_scale.fb).transpose(1, 2)
        return mel.clamp_min(1e-7).log()


class VocosMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Features()
        self.backbone = Backbone()
        self.head = Head()

    @classmethod
    def pretrained(cls):
        path = hf_hub_download(MODEL, "pytorch_model.bin", revision=REVISION)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls()
        model.load_state_dict(state, strict=True)
        return model.eval()

    def forward(self, x):
        return self.head(self.backbone(self.feature_extractor(x)))
