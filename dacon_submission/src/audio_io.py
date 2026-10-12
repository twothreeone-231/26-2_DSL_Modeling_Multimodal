"""Robust audio I/O helpers shared by the presence detector and separator.

Handles: WAV/FLAC/MP3, mono/stereo, 16k/44.1k/48k, very short audio, silence,
NaN/Inf validation, clipping detection. Resampling for the separator uses
torchaudio.functional.resample (GPU/torch-friendly) rather than
librosa.resample, per the task spec.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".wma"}

# soundfile/libsndfile (this env: 1.2.0) can decode WAV/FLAC/MP3/OGG but NOT
# m4a/aac/opus/wma -- and torchaudio here has no ffmpeg backend registered
# (torchaudio.list_audio_backends() == ["soundfile"], same libsndfile
# limitation underneath, confirmed empirically) and there is no system
# `ffmpeg` binary on PATH either. Without a third decoder, any of these 4
# extensions -- which SUPPORTED_AUDIO_EXTENSIONS above claims to support --
# would crash load_audio() outright. `av` (PyAV) ships its own statically
# linked FFmpeg libraries in the wheel, so it needs no system ffmpeg binary
# and works fully offline once installed, including inside a DACON submit.zip
# (see requirements.txt / scripts/build_submission.py).
# (kept only for documentation/tests -- load_audio() below always tries
# soundfile -> torchaudio -> av in order regardless of extension, since the
# actual codec inside a container can't be inferred from the extension alone)


@dataclass
class AudioInfo:
    path: str
    sample_rate: int
    num_channels: int
    num_samples: int
    duration_seconds: float
    is_clipped: bool
    peak_abs: float
    rms: float


class AudioLoadError(ValueError):
    pass


def validate_finite(waveform: np.ndarray, source: str = "audio") -> None:
    if not np.isfinite(waveform).all():
        raise AudioLoadError(f"{source} contains NaN or Inf samples")


def detect_clipping(waveform: np.ndarray, threshold: float = 0.999) -> bool:
    if waveform.size == 0:
        return False
    return bool(np.abs(waveform).max() >= threshold)


def compute_rms(waveform: np.ndarray) -> float:
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def rms_to_db(rms: float, eps: float = 1e-10) -> float:
    return float(20.0 * np.log10(max(rms, eps)))


def _load_via_pyav(path: str) -> Tuple[np.ndarray, int]:
    """Decode via PyAV's bundled FFmpeg libraries (no system ffmpeg needed).

    Returns [T, C] float32 (to match the sf.read(..., always_2d=True)
    convention the caller expects), at the stream's native sample rate.
    """
    import av

    container = av.open(path)
    if not container.streams.audio:
        raise AudioLoadError(f"No audio stream found in {path}")
    stream = container.streams.audio[0]
    sr = stream.rate

    # Force a single, known float output layout regardless of the source
    # codec's native sample format (int16/int32/planar/interleaved/etc.) --
    # av.audio.resampler.AudioResampler handles that conversion uniformly so
    # this function never has to special-case per-codec sample formats.
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sr)
    chunks = []
    for frame in container.decode(stream):
        for rframe in resampler.resample(frame):
            chunks.append(rframe.to_ndarray())  # [C, T] float32 planar
    container.close()

    if not chunks:
        raise AudioLoadError(f"PyAV decoded zero audio frames from {path}")
    waveform_ct = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)  # [C, T]
    return waveform_ct.T, sr  # -> [T, C]


def load_audio(path: str | Path, mono: bool = False) -> Tuple[np.ndarray, int]:
    """Load an audio file as float32 waveform.

    Returns:
        waveform: shape [T] if mono else [C, T] (channels-first, torchaudio convention)
        sample_rate: native sample rate of the file (no resampling here)
    """
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        logger.warning("Unrecognized audio extension %s for %s; attempting to load anyway", path.suffix, path)

    try:
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
    except Exception as exc_sf:  # noqa: BLE001 - soundfile/libsndfile lacks m4a/aac/opus/wma
        logger.info("soundfile failed to read %s (%s); falling back to torchaudio", path, exc_sf)
        try:
            wav, sr = torchaudio.load(str(path))  # [C, T]
            waveform = wav.numpy().T  # -> [T, C]
        except Exception as exc_ta:  # noqa: BLE001 - torchaudio has no ffmpeg backend in this env either
            logger.info("torchaudio failed to read %s (%s); falling back to PyAV", path, exc_ta)
            waveform, sr = _load_via_pyav(str(path))  # [T, C]

    if waveform.size == 0:
        raise AudioLoadError(f"Audio file is empty: {path}")

    waveform = waveform.T.astype(np.float32, copy=False)  # [C, T]
    validate_finite(waveform, source=str(path))

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0, keepdims=True)
    if mono:
        waveform = waveform[0]

    return waveform, int(sr)


def get_audio_info(path: str | Path) -> AudioInfo:
    waveform, sr = load_audio(path, mono=False)
    num_channels, num_samples = waveform.shape
    return AudioInfo(
        path=str(path),
        sample_rate=sr,
        num_channels=num_channels,
        num_samples=num_samples,
        duration_seconds=num_samples / sr if sr > 0 else 0.0,
        is_clipped=detect_clipping(waveform),
        peak_abs=float(np.abs(waveform).max()) if waveform.size else 0.0,
        rms=compute_rms(waveform),
    )


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """[C, T] -> [1, T] by averaging channels. No-op if already mono."""
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """[1 or 2, T] -> [2, T]. Mono is duplicated to both channels."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        return waveform.repeat(2, 1)
    if waveform.shape[0] == 2:
        return waveform
    # More than 2 channels: downmix to mono first, then duplicate.
    return to_stereo(to_mono(waveform))


def resample_torch(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """GPU/torch-friendly resampling via torchaudio.functional.resample."""
    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)


def fit_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Crop or zero-pad the last (time) dim of [..., T] to exactly target_length."""
    cur_length = waveform.shape[-1]
    if cur_length == target_length:
        return waveform
    if cur_length > target_length:
        return waveform[..., :target_length]
    pad = target_length - cur_length
    return torch.nn.functional.pad(waveform, (0, pad))
