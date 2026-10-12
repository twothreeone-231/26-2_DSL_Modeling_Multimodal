"""AudioPresenceDetector: sliding-window AudioMosaic inference -> Voice/Music
probabilities, using the *official* AudioMosaic preprocessing implementation
(vendored inside the hanxunh/AudioMosaic-vit-b16-finetune-as2m HF repo), not
a reimplementation.

See AUDIO_PRESENCE_SEPARATION_TASK.md sections 3-6.
"""
from __future__ import annotations

import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torchaudio

from . import audioset_labels, config

logger = logging.getLogger(__name__)


def _aggregate(values: np.ndarray, method: str, top_k_ratio: float, percentile: float) -> float:
    """Collapse an array of scores into one score. Never a plain mean/sum over
    many classes/windows (that biases toward groups with more members / more
    windows) — see task doc section 6."""
    if values.size == 0:
        return 0.0
    if method == "max":
        return float(values.max())
    if method == "percentile":
        return float(np.percentile(values, percentile))
    if method == "topk_mean":
        k = max(1, int(round(values.size * top_k_ratio)))
        k = min(k, values.size)
        top_k = np.partition(values, -k)[-k:]
        return float(top_k.mean())
    raise ValueError(f"Unknown aggregation method: {method}")


class AudioPresenceDetector:
    """Voice/Music presence + confidence via sliding-window AudioMosaic inference."""

    def __init__(
        self,
        cfg: Optional[config.PresenceConfig] = None,
        device: Optional[torch.device] = None,
        hf_cache_dir: Path = config.DEFAULT_HF_CACHE_DIR,
        model_id: str = config.AUDIO_MOSAIC_MODEL,
    ) -> None:
        self.cfg = cfg or config.PresenceConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.ensure_hf_cache_env(hf_cache_dir)

        t0 = time.time()
        from huggingface_hub import snapshot_download

        logger.info("Loading AudioMosaic model '%s' (device=%s)...", model_id, self.device)
        self._repo_dir = snapshot_download(model_id)
        if self._repo_dir not in sys.path:
            sys.path.insert(0, self._repo_dir)
        from load_model import load_classifier, preprocess as official_preprocess  # type: ignore

        self._preprocess_fn = official_preprocess
        self.model = load_classifier(repo_dir=self._repo_dir, device=str(self.device)).eval()
        logger.info("AudioMosaic loaded in %.2fs", time.time() - t0)

        import json

        with open(Path(self._repo_dir) / "config.json") as f:
            model_cfg = json.load(f)
        self.audio_preprocessing_cfg: Dict = model_cfg["audio_preprocessing"]

        self.ontology = audioset_labels.validate_label_groups()
        self.voice_indices = audioset_labels.label_indices(self.cfg.voice_labels_or_default(), self.ontology)
        self.music_indices = audioset_labels.label_indices(self.cfg.music_labels_or_default(), self.ontology)
        logger.info(
            "Voice label group: %d classes, Music label group: %d classes",
            len(self.voice_indices),
            len(self.music_indices),
        )

        self.window_seconds, self.hop_seconds = self._resolve_window_hop()
        logger.info(
            "Sliding window: window=%.3fs hop=%.3fs (derived from AudioMosaic target_length=%d, frame_shift=%.1fms)",
            self.window_seconds,
            self.hop_seconds,
            self.audio_preprocessing_cfg["target_length"],
            self.audio_preprocessing_cfg["frame_shift"],
        )

    # ------------------------------------------------------------------
    def _resolve_window_hop(self) -> tuple[float, float]:
        if self.cfg.window_seconds is not None:
            window_seconds = self.cfg.window_seconds
        else:
            sr = self.audio_preprocessing_cfg["sample_rate"]
            target_length = self.audio_preprocessing_cfg["target_length"]
            frame_shift_ms = self.audio_preprocessing_cfg["frame_shift"]
            # frame_length is not overridden by AudioMosaic; read torchaudio's
            # own default so we do not guess/hardcode kaldi's 25ms convention.
            frame_length_ms = inspect.signature(torchaudio.compliance.kaldi.fbank).parameters["frame_length"].default
            frame_shift_samples = round(sr * frame_shift_ms / 1000.0)
            frame_length_samples = round(sr * frame_length_ms / 1000.0)
            window_samples = (target_length - 1) * frame_shift_samples + frame_length_samples
            window_seconds = window_samples / sr
        hop_seconds = self.cfg.hop_seconds if self.cfg.hop_seconds is not None else window_seconds / 2.0
        return window_seconds, hop_seconds

    def _make_windows(self, num_samples: int, sr: int) -> List[tuple[int, int]]:
        window_len = max(1, int(round(self.window_seconds * sr)))
        hop_len = max(1, int(round(self.hop_seconds * sr)))
        if num_samples <= window_len:
            return [(0, num_samples)]
        windows = []
        start = 0
        while start < num_samples:
            end = min(start + window_len, num_samples)
            windows.append((start, end))
            if end == num_samples:
                break
            start += hop_len
        # Make sure the tail is covered by a full-length window too (official
        # policy is to pad the last window, not to drop leftover audio).
        last_start, last_end = windows[-1]
        if last_end - last_start < window_len and last_end == num_samples and last_start != 0:
            tail_start = max(0, num_samples - window_len)
            if (tail_start, num_samples) != (last_start, last_end):
                windows.append((tail_start, num_samples))
        return windows

    @torch.inference_mode()
    def _score_windows(self, waveform: torch.Tensor, sr: int, windows: Sequence[tuple[int, int]]) -> np.ndarray:
        """Returns array [num_windows, 527] of sigmoid probabilities."""
        batch_size = self.cfg.batch_size
        all_probs = []
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i : i + batch_size]
            fbanks = []
            for start, end in batch_windows:
                segment = waveform[:, start:end]
                fbank = self._preprocess_fn((segment, sr), repo_dir=self._repo_dir)
                fbanks.append(fbank)
            batch = torch.cat(fbanks, dim=0).to(self.device)
            logits = self.model(batch)
            probs = logits.sigmoid().float().cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def predict(self, waveform, sample_rate: int) -> Dict:
        """Run sliding-window AudioMosaic inference on a waveform.

        Args:
            waveform: numpy array [T] or [C, T], or torch tensor, any sample rate.
            sample_rate: sample rate of `waveform`.
        """
        t0 = time.time()
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.float()

        num_samples = waveform.shape[-1]
        windows = self._make_windows(num_samples, sample_rate)
        window_probs = self._score_windows(waveform, sample_rate, windows)  # [W, 527]

        voice_window_scores = np.array(
            [
                _aggregate(
                    window_probs[w, self.voice_indices],
                    self.cfg.window_class_aggregation,
                    self.cfg.window_top_k_ratio,
                    self.cfg.window_percentile,
                )
                for w in range(len(windows))
            ]
        )
        music_window_scores = np.array(
            [
                _aggregate(
                    window_probs[w, self.music_indices],
                    self.cfg.window_class_aggregation,
                    self.cfg.window_top_k_ratio,
                    self.cfg.window_percentile,
                )
                for w in range(len(windows))
            ]
        )

        voice_probability = _aggregate(voice_window_scores, self.cfg.aggregation, self.cfg.top_k_ratio, self.cfg.percentile)
        music_probability = _aggregate(music_window_scores, self.cfg.aggregation, self.cfg.top_k_ratio, self.cfg.percentile)

        voice_present = voice_probability >= self.cfg.voice_threshold
        music_present = music_probability >= self.cfg.music_threshold

        windows_out = []
        for w, (start, end) in enumerate(windows):
            windows_out.append(
                {
                    "start": start / sample_rate,
                    "end": end / sample_rate,
                    "voice_score": float(voice_window_scores[w]),
                    "music_score": float(music_window_scores[w]),
                }
            )
            logger.debug(
                "window[%d] %.2f-%.2fs voice=%.4f music=%.4f",
                w,
                start / sample_rate,
                end / sample_rate,
                voice_window_scores[w],
                music_window_scores[w],
            )

        elapsed = time.time() - t0
        logger.info(
            "AudioMosaic: %d windows, voice=%.4f music=%.4f (%.3fs)",
            len(windows),
            voice_probability,
            music_probability,
            elapsed,
        )

        return {
            "voice_probability": voice_probability,
            "music_probability": music_probability,
            "voice_present": bool(voice_present),
            "music_present": bool(music_present),
            "raw": {
                "num_windows": len(windows),
                "window_seconds": self.window_seconds,
                "hop_seconds": self.hop_seconds,
                "aggregation": self.cfg.aggregation,
                "window_class_aggregation": self.cfg.window_class_aggregation,
                "inference_time": elapsed,
            },
            "windows": windows_out,
        }

    def predict_file(self, path: str | Path) -> Dict:
        from . import audio_io

        waveform, sr = audio_io.load_audio(path, mono=False)
        return self.predict(waveform, sr)
