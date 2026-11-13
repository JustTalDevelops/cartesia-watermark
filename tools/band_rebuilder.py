"""Speech-band inpainting with no access to the input's removed frequency band."""
import torch
from torch import nn
from torch.nn import functional as F


class BandRebuilder(nn.Module):
    def __init__(self, low=1500., high=3500., shoulder=100., width=16, mix=1., min_snr_db=None,
                 noise_input=False, post_lowpass_hz=None, phase_hint=False,
                 phase_fft_size=512, magnitude_blur=2., magnitude_time_blur=0.):
        super().__init__()
        self.config=dict(low=low,high=high,shoulder=shoulder,width=width,mix=mix,min_snr_db=min_snr_db,
                         noise_input=noise_input,post_lowpass_hz=post_lowpass_hz,
                         phase_hint=phase_hint,phase_fft_size=phase_fft_size,
                         magnitude_blur=magnitude_blur,magnitude_time_blur=magnitude_time_blur)
        self.register_buffer('phase_window',torch.hann_window(phase_fft_size),persistent=False)
        channels=[width,width*2,width*4,width*6]
        self.first=nn.Conv1d(2 if noise_input else 1,channels[0],15,padding=7)
        self.down=nn.ModuleList([nn.Conv1d(a,b,8,stride=4,padding=2) for a,b in zip(channels[:-1],channels[1:])])
        self.context=nn.ModuleList([nn.Conv1d(channels[-1],channels[-1],3,padding=d,dilation=d) for d in (1,2,4,8,16)])
        self.up=nn.ModuleList([nn.Conv1d(a+b,b,5,padding=2) for a,b in zip(channels[:0:-1],channels[-2::-1])])
        self.last=nn.Conv1d(channels[0],1,15,padding=7)
        nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)
        if noise_input:
            nn.init.normal_(self.last.weight,std=.005)

    def band(self,x):
        f=torch.fft.rfftfreq(x.shape[-1],d=1/16000,device=x.device)
        a=((f-self.config['low']+self.config['shoulder'])/self.config['shoulder']).clamp(0,1)
        b=((self.config['high']+self.config['shoulder']-f)/self.config['shoulder']).clamp(0,1)
        return torch.fft.irfft(torch.fft.rfft(x)*(a*b),n=x.shape[-1])

    def forward(self,x,strength=1.):
        if x.shape[-1]<1024:
            return x.clone()
        outside=x-self.band(x)
        scale=outside.square().mean(-1,keepdim=True).clamp_min(1e-10).sqrt()
        features=(outside/scale)[:,None]
        if self.config['noise_input']:
            if self.training:
                noise=torch.randn_like(features)
            else:
                generator=torch.Generator(device=x.device).manual_seed(2046)
                noise=torch.randn(features.shape,generator=generator,device=x.device,dtype=x.dtype)
            features=torch.cat((features,noise),1)
        h=F.silu(self.first(features));skips=[h]
        for layer in self.down:
            h=F.silu(layer(h));skips.append(h)
        for layer in self.context:
            h=h+F.silu(layer(h))
        for layer,skip in zip(self.up,skips[-2::-1]):
            h=F.interpolate(h,size=skip.shape[-1],mode='linear',align_corners=False)
            h=F.silu(layer(torch.cat((h,skip),1)))
        prediction=self.last(h)[:,0].tanh()*scale*.5
        rebuilt=outside+self.band(prediction)
        if self.config['phase_hint']:
            size=self.config['phase_fft_size']
            original=torch.stft(x,size,size//4,window=self.phase_window,return_complex=True)
            clean=torch.stft(rebuilt,size,size//4,window=self.phase_window,return_complex=True)
            magnitude=original.abs()[:,None]
            for axis,sigma in ((2,self.config['magnitude_blur']),(3,self.config['magnitude_time_blur'])):
                if sigma:
                    radius=max(1,round(3*sigma))
                    index=torch.arange(-radius,radius+1,device=x.device,dtype=x.dtype)
                    kernel=torch.exp(-.5*(index/sigma).square());kernel=kernel/kernel.sum()
                    padding=(0,0,radius,radius) if axis==2 else (radius,radius,0,0)
                    shape=(1,1,-1,1) if axis==2 else (1,1,1,-1)
                    magnitude=F.conv2d(F.pad(magnitude,padding,mode='replicate'),kernel.reshape(shape))
            f=torch.fft.rfftfreq(size,1/16000,device=x.device)
            mask=((f-self.config['low']+self.config['shoulder'])/self.config['shoulder']).clamp(0,1)
            mask=mask*((self.config['high']+self.config['shoulder']-f)/self.config['shoulder']).clamp(0,1)
            spectrum=original*(1-mask[None,:,None])+torch.polar(magnitude[:,0],clean.angle())*mask[None,:,None]
            rebuilt=torch.istft(spectrum,size,size//4,window=self.phase_window,length=x.shape[-1])
        delta=(rebuilt-x)*self.config['mix']*strength
        if self.config['min_snr_db'] is not None:
            budget=x.square().mean(-1,keepdim=True).clamp_min(1e-20).sqrt()*10**(-self.config['min_snr_db']/20)*min(strength,1.)
            delta=delta*(budget/delta.square().mean(-1,keepdim=True).clamp_min(1e-24).sqrt()).clamp(max=1)
        return x+delta
