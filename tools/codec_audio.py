"""Complete-waveform operations for the experimental codec candidate."""
import hashlib
import numpy as np
from scipy.signal import butter, sosfiltfilt
import soundfile as sf


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_audio(audio, rate):
    if (audio.ndim not in (1, 2) or (audio.ndim == 2 and audio.shape[1] not in (1, 2)) or
            not np.isfinite(audio).all() or not 8000 <= rate <= 96000 or
            not 5*rate <= len(audio) <= 300*rate):
        raise ValueError('Choose complete finite mono/stereo speech, 5–300 seconds at 8–96 kHz')


def validate_paths(source, output, metadata):
    if source.resolve() == output.resolve() or output.exists() or output.suffix.lower() != '.wav':
        raise ValueError('Choose a new WAV output separate from the input')
    if metadata and (metadata.exists() or metadata.resolve() in (source.resolve(), output.resolve())):
        raise ValueError('Choose a new metadata path separate from the audio files')


def quantize(path, audio, rate):
    if not np.isfinite(audio).all():
        raise RuntimeError('Cannot encode nonfinite samples')
    sf.write(path, audio, rate, format='FLAC', subtype='PCM_24')
    decoded, actual_rate = sf.read(path, dtype='float32')
    if (actual_rate != rate or decoded.shape != audio.shape or not np.isfinite(decoded).all() or
            np.max(np.abs(decoded-audio)) > 2**-23):
        raise RuntimeError('PCM24 encoding changed the format or exceeded its quantization limit')
    return decoded


def mix_reconstructions(vocoder, codec, sample_rate, low=1000, high=4000):
    """Runtime counterpart of the separately retained research reference."""
    vocoder, codec = np.asarray(vocoder), np.asarray(codec)
    validate_audio(vocoder, sample_rate)
    validate_audio(codec, sample_rate)
    if vocoder.shape != codec.shape:
        raise ValueError('Reconstructions must have the same complete sample grid')
    if not 0 < low < high <= sample_rate/2:
        raise ValueError('Band must lie within the available frequencies')
    vocoder, codec = np.asarray(vocoder, dtype=np.float32), np.asarray(codec, dtype=np.float32)
    # At 8 kHz, the fixed 4 kHz upper edge includes every available frequency
    # above the lower edge; its realizable filter is a high pass.
    sos = (butter(6, low, btype='highpass', fs=sample_rate, output='sos') if high == sample_rate/2 else
           butter(6, [low, high], btype='bandpass', fs=sample_rate, output='sos'))
    output = np.asarray(vocoder+sosfiltfilt(sos, codec-vocoder, axis=0), dtype=np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError('Band replacement produced nonfinite samples')
    gain = min(1., .999/max(1e-12, float(np.max(np.abs(output)))))
    return output*gain, gain
