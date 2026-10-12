"""LofczMusicFakeDetector: AI-generated-music probability via
lofcz/ai-music-detector (MIT license, github.com/lofcz/ai-music-detector) --
a tiny logistic-regression classifier on a hand-crafted spectral
"fakeprint" feature (regularly-spaced 1-8kHz peaks left by transposed-
convolution vocoders), targeting Suno<=5/Udio<=1.5 specifically. No
training performed by us; the fakeprint algorithm and weights are
reproduced/loaded verbatim from the author's own inference.py /
extract_fakeprints.py (src/python/ in the upstream repo) and
model.safetensors -- see model/lofcz_ai_music_detector/README.md.

Reopened as a music-fake-detection candidate alongside ArtifactNet after a
direct comparison on the canonical 100-file proxy set
(outputs/artifactnet_eval/predictions_gpu_patched.csv, FakeMusicCaps fake +
real music, 50/50) showed better *ranking* quality than ArtifactNet despite
zero exposure to any of FakeMusicCaps' 5 generators (MusicGen, AudioLDM2,
MusicLDM, Mustango, Stable Audio Open): EER=24.00%/AUC=0.8128 vs
ArtifactNet's EER=30.0%/AUC=0.766 (jaehyun/musicdet/evaluate_lofcz.py).
Accuracy@0.5 on that same set is a near-random 52%, because the model's
native 0.5 decision threshold was calibrated against Suno/Udio's specific
artifact strength, not FakeMusicCaps' generators -- this module
deliberately keeps that native threshold unrecalibrated (see
config.LofczMusicFakeConfig), since our proxy set may not represent
DACON's actual (unknown) generator mix either.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torchaudio
from scipy.ndimage import minimum_filter1d

from . import audio_io, config

logger = logging.getLogger(__name__)


class LofczMusicFakeDetector:
    """AI-generated-music probability via lofcz/ai-music-detector."""

    model_id = "lofcz/ai-music-detector"
    model_version = "master"

    def __init__(
        self,
        cfg: Optional[config.LofczMusicFakeConfig] = None,
        device: Optional[torch.device] = None,
        hf_cache_dir: Path = config.DEFAULT_HF_CACHE_DIR,
    ) -> None:
        self.cfg = cfg or config.LofczMusicFakeConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = Path(self.cfg.model_dir) if self.cfg.model_dir else config.LOFCZ_MODEL_DIR
        t0 = time.time()

        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            model_cfg = json.load(f)
        pp = model_cfg["preprocessing"]
        self.sample_rate = pp["sample_rate"]
        self.n_fft = pp["n_fft"]
        self.freq_min = pp["freq_min"]
        self.freq_max = pp["freq_max"]
        self.hull_area = pp["hull_area"]
        self.max_db = pp["max_db"]
        self.min_db = pp["min_db"]
        self.max_duration = 300  # matches upstream inference.py's own cap

        from safetensors.numpy import load_file

        weights = load_file(str(model_dir / "model.safetensors"))
        self.w = weights["weights"]  # (1, F)
        self.b = weights["bias"]  # (1,)
        self.expected_features = self.w.shape[1]

        use_gpu = self.cfg.prefer_gpu and self.device.type == "cuda"
        self.stft_device = self.device if use_gpu else torch.device("cpu")
        self.stft = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, power=2, normalized=False).to(self.stft_device)
        self.freq_bins = np.linspace(0, self.sample_rate / 2, num=(self.n_fft // 2) + 1)
        self.freq_mask = (self.freq_bins >= self.freq_min) & (self.freq_bins <= self.freq_max)

        logger.info(
            "LofczMusicFakeDetector loaded from %s in %.2fs (feature_dim=%d, device=%s, threshold=%.2f)",
            model_dir, time.time() - t0, self.expected_features, self.stft_device, self.cfg.predicted_label_threshold,
        )

    # ------------------------------------------------------------------
    def _compute_fakeprint(self, mono_waveform: torch.Tensor) -> np.ndarray:
        """Reproduces lofcz's inference.py AIDetector.compute_fakeprint()
        exactly: mean dB power spectrum -> 1-8kHz mask -> lower-hull
        subtraction (isolates deconvolution artifact peaks from melodic
        content) -> clip -> max-normalize."""
        audio = mono_waveform.unsqueeze(0).to(self.stft_device)  # [1, T]
        with torch.no_grad():
            spec = self.stft(audio)
        spec_db = 10 * torch.log10(torch.clamp(spec, min=1e-10, max=1e6))
        mean_spectrum = spec_db.mean(dim=(0, 2)).cpu().numpy()
        freq_spectrum = mean_spectrum[self.freq_mask]
        hull = minimum_filter1d(freq_spectrum, size=self.hull_area, mode="nearest")
        hull = np.clip(hull, self.min_db, None)
        residue = np.clip(freq_spectrum - hull, 0, None)
        residue = np.clip(residue, 0, self.max_db)
        max_val = np.max(residue) + 1e-6
        return (residue / max_val).astype(np.float32)

    def predict(self, waveform, sample_rate: int) -> Dict:
        """Run lofcz/ai-music-detector on a waveform.

        Args:
            waveform: numpy array or torch tensor. 1D [T] mono, or 2D [C, T]
                channels-first (torchaudio convention, matches audio_io.load_audio).
            sample_rate: sample rate of `waveform`.
        """
        t0 = time.time()
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        waveform = waveform.float()

        if waveform.dim() == 1:
            channels_first = waveform.unsqueeze(0)  # [1, T]
        elif waveform.dim() == 2:
            channels_first = waveform  # assumed [C, T]
        else:
            raise ValueError(f"LofczMusicFakeDetector expects a 1D [T] or 2D [C, T] waveform, got shape {tuple(waveform.shape)}")

        if channels_first.shape[-1] == 0:
            raise ValueError("Cannot run LofczMusicFakeDetector on an empty waveform")

        mono = audio_io.to_mono(channels_first)  # [1, T]

        t_resample0 = time.time()
        resampled = audio_io.resample_torch(mono, sample_rate, self.sample_rate)[0]  # [T] @ 16000
        resample_time = time.time() - t_resample0

        max_samples = self.max_duration * self.sample_rate
        if resampled.shape[-1] > max_samples:
            resampled = resampled[:max_samples]

        resampled_np = resampled.numpy()
        audio_io.validate_finite(resampled_np, "LofczMusicFakeDetector input audio")
        duration = resampled.shape[-1] / self.sample_rate

        t_infer0 = time.time()
        fakeprint = self._compute_fakeprint(resampled)
        if len(fakeprint) != self.expected_features:
            old_x = np.linspace(0, 1, len(fakeprint))
            new_x = np.linspace(0, 1, self.expected_features)
            fakeprint = np.interp(new_x, old_x, fakeprint).astype(np.float32)
        raw_logit = float(np.dot(fakeprint, self.w.T).reshape(-1)[0] + self.b[0])
        raw_probability = float(1.0 / (1.0 + np.exp(-raw_logit)))
        calibrated_logit = self.cfg.calibration_scale * raw_logit + self.cfg.calibration_bias
        fake_probability = float(1.0 / (1.0 + np.exp(-calibrated_logit)))
        infer_time = time.time() - t_infer0

        real_probability = 1.0 - fake_probability
        predicted_label = "ai_generated" if fake_probability > self.cfg.predicted_label_threshold else "human_real"

        total_time = time.time() - t0

        # Single whole-file score (no segmentation, unlike ArtifactNet) --
        # "segments"/"stats" below are trivial 1-element structures kept only
        # so this detector is a drop-in for pipeline.py's ArtifactNet-shaped
        # diagnostics/logging code.
        return {
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "predicted_label": predicted_label,
            "segment_probabilities": [fake_probability],
            "aggregation": "whole_file_fakeprint",
            "num_segments": 1,
            "num_valid_segments": 1,
            "num_invalid_segments": 0,
            "input_duration": duration,
            "model": self.model_id,
            "model_version": self.model_version,
            "raw_probability": raw_probability,
            "calibration": {"scale": self.cfg.calibration_scale, "bias": self.cfg.calibration_bias},
            "segments": [
                {
                    "start": 0.0,
                    "end": duration,
                    "valid_duration": duration,
                    "padded": False,
                    "fake_probability": fake_probability,
                    "valid": True,
                }
            ],
            "stats": {
                "median": fake_probability,
                "mean": fake_probability,
                "std": 0.0,
                "min": fake_probability,
                "max": fake_probability,
            },
            "timings": {
                "resample": resample_time,
                "inference": infer_time,
                "total": total_time,
            },
        }

    def predict_file(self, path: str | Path) -> Dict:
        t0 = time.time()
        waveform, sr = audio_io.load_audio(path, mono=False)
        load_time = time.time() - t0
        result = self.predict(waveform, sr)
        result["timings"]["audio_load"] = load_time
        result["timings"]["total"] += load_time
        return result
