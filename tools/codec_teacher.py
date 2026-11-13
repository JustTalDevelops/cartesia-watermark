"""Differentiable mel reconstruction to reject fragile phase-only edits."""
import torch
from torch import nn
from torch.nn import functional as F
from third_party.vocos_mel import VocosMel


def fft_resample(x,n):
    old=x.shape[-1]
    spectrum=torch.fft.rfft(x)
    count=n//2+1
    spectrum=F.pad(spectrum[...,:count],(0,max(0,count-spectrum.shape[-1])))
    return torch.fft.irfft(spectrum,n=n)*(n/old)


class MelReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocoder=VocosMel.pretrained().requires_grad_(False).eval()

    def forward(self,x):
        scale=x.square().mean(-1,keepdim=True).clamp_min(1e-10).sqrt()
        wave24=fft_resample(x*.1/scale,round(x.shape[-1]*1.5))
        rebuilt=self.vocoder(wave24)
        rebuilt=F.pad(rebuilt[...,:wave24.shape[-1]],(0,max(0,wave24.shape[-1]-rebuilt.shape[-1])))
        return fft_resample(rebuilt,x.shape[-1])*scale/.1


class NativeMelReconstruction(nn.Module):
    """24 kHz input/output; preserve the vocoder's full native input bandwidth."""
    def __init__(self):
        super().__init__()
        self.vocoder = VocosMel.pretrained().requires_grad_(False).eval()

    def forward(self, x):
        scale = x.square().mean(-1, keepdim=True).clamp_min(1e-10).sqrt()
        rebuilt = self.vocoder(x*.1/scale)
        rebuilt = F.pad(rebuilt[..., :x.shape[-1]],
                        (0, max(0, x.shape[-1]-rebuilt.shape[-1])))
        return rebuilt*scale/.1
