"""VoiceFakeDetector: wraps DF-Arena 1B (Speech-Arena-2025/DF_Arena_1B_V_1,
wav2vec2-XLS-R-1B + Conformer voice/speech antispoofing) as this pipeline's
voice/speech deepfake detector.

Reverted back from WavLM (sewon/wavlm_module) after a real, measured
comparison: on the exact same 100 (voice, music) pairs (real inference,
BS PolarFormer separation), DF-Arena scored clean EER=4.00%/AUC=0.9984 ->
separated-vocal EER=13.00%/AUC=0.9028 (+9pp), while WavLM scored clean
EER=11.00%/AUC=0.9102 -> separated-vocal EER=32.00%/AUC=0.7580 (+21pp).
DF-Arena is both more accurate on clean audio and far more robust to BS
PolarFormer's separation artifacts -- see README's "Voice deepfake
detection" section and SUBMISSION_LOG.md for the real leaderboard
regression this explains (submission 1, DF-Arena: ADS=0.6429; submissions
2-3, WavLM: ADS=0.5708-0.5721, despite the WavLM swap looking neutral or
positive on isolated clean-audio benchmarks).

Segment extraction / aggregation logic (SEGMENT_SAMPLES, tile-repeat for
short clips, silence gating, max-over-segments) is reused verbatim from
../baseline/script.py's own DF-Arena usage -- not reimplemented/guessed.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torchaudio

from . import audio_io, config

logger = logging.getLogger(__name__)

MODEL_LABEL = "Speech-Arena-2025/DF_Arena_1B_V_1"
MODEL_SAMPLE_RATE = 16000


def _allow_torch_load_below_2_6() -> None:
    """This dev env's torch (2.5.1) is older than transformers' CVE-2025-32434
    gate for .bin checkpoints; the real DACON eval server's torch (2.7.1)
    never needs this. `check_torch_load_is_safe()` is imported *by name*
    into transformers.modeling_utils's own namespace and raises
    unconditionally (regardless of the `weights_only` kwarg) when torch<2.6,
    so it must be patched there -- patching torch.load itself, or patching
    transformers.utils.import_utils.check_torch_load_is_safe (where it's
    merely *defined*), does not stop the raise (confirmed empirically)."""
    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if (major, minor) < (2, 6):
        import transformers.modeling_utils as _mu

        _mu.check_torch_load_is_safe = lambda: None


def _get_segment_starts(audio_length: int, segment_samples: int) -> list[int]:
    if audio_length <= segment_samples:
        return [0]
    last_start = audio_length - segment_samples
    starts = list(range(0, last_start + 1, segment_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _extract_segment(audio: np.ndarray, start: int, segment_samples: int) -> np.ndarray:
    if audio.size < segment_samples:
        repeat_count = segment_samples // audio.size + 1
        tiled = np.tile(audio, repeat_count)
        return tiled[:segment_samples].astype(np.float32)
    end = start + segment_samples
    return audio[start:end].astype(np.float32, copy=False)


class VoiceFakeDetector:
    """Voice/speech deepfake probability via DF-Arena 1B."""

    def __init__(
        self,
        cfg: Optional[config.VoiceFakeDetectorConfig] = None,
        device: Optional[torch.device] = None,
        hf_cache_dir: Path = config.DEFAULT_HF_CACHE_DIR,
    ) -> None:
        self.cfg = cfg or config.VoiceFakeDetectorConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Must happen before anything imports transformers/huggingface_hub:
        # huggingface_hub bakes its cache-path constants in at first import
        # time, so calling this after that import (e.g. only inside another
        # detector's constructor) is too late on a SLURM compute node (only
        # /mnt/data1/... is writable there, not /data1/...). See README
        # section 15 -- this exact bug was hit and fixed for this file once
        # already (WavLM era) and again while re-validating DF-Arena.
        config.ensure_hf_cache_env(hf_cache_dir)
        _allow_torch_load_below_2_6()

        model_dir = Path(self.cfg.module_dir) if self.cfg.module_dir else config.DF_ARENA_MODEL_DIR
        if not model_dir.is_dir():
            raise FileNotFoundError(f"DF-Arena model dir not found: {model_dir}")

        t0 = time.time()
        model_dir_str = str(model_dir.parent)
        if model_dir_str not in sys.path:
            sys.path.insert(0, model_dir_str)
        from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing  # type: ignore

        use_gpu = self.cfg.prefer_gpu and self.device.type == "cuda"
        load_device = torch.device("cuda" if use_gpu else "cpu")

        previous_directory = Path.cwd()
        try:
            os.chdir(model_dir)
            self._model = DF_Arena_1B_Antispoofing.from_pretrained(
                str(model_dir), local_files_only=True, low_cpu_mem_usage=True,
            )
        finally:
            os.chdir(previous_directory)
        self._model = self._model.to(load_device).eval()
        self._load_device = load_device
        self._fake_label_index = int(self._model.config.label2id["spoof"])
        # Component-isolation probe (2026-09-27): an optional fine-tuned copy of
        # the Conformer scores the SAME Conformer input as the published one;
        # its result is only stashed on the detector for script.py. predict()'s
        # return value -- and FILE_FAKE_PROB built from it -- is unchanged.
        self.last_side_probability = None
        self._side_conformer = _load_side_conformer(self._model, load_device)
        if self._side_conformer is not None:
            self._side_in = {}
            self._model.backbone.conformer.register_forward_pre_hook(
                lambda _m, inp: self._side_in.__setitem__("x", inp[0]))

        logger.info(
            "DF-Arena voice-fake detector loaded in %.2fs (device=%s, model_dir=%s)",
            time.time() - t0, load_device, model_dir,
        )

    def predict(self, waveform, sample_rate: int) -> Dict:
        t0 = time.time()
        if isinstance(waveform, np.ndarray):
            waveform_t = torch.from_numpy(waveform).float()
        else:
            waveform_t = waveform.float()

        mono = audio_io.to_mono(waveform_t).squeeze(0) if waveform_t.dim() == 2 else waveform_t
        if sample_rate != MODEL_SAMPLE_RATE:
            mono = torchaudio.functional.resample(mono, sample_rate, MODEL_SAMPLE_RATE)
        audio_np = mono.numpy().astype(np.float32, copy=False)

        rms = float(np.sqrt(np.mean(np.square(audio_np, dtype=np.float64))))
        if rms < self.cfg.silence_rms_floor:
            elapsed = time.time() - t0
            logger.info("DF-Arena: silent input (rms=%.2e), skipping inference", rms)
            return {
                "fake_probability": 0.0,
                "real_probability": 1.0,
                "predicted_label": "real",
                "all_silent": True,
                "model": MODEL_LABEL,
                "timings": {"total": elapsed},
            }

        # Per-segment silence gating is REQUIRED, not just the whole-file RMS
        # check above: DF-Arena returns a fixed, content-independent score
        # (~0.7642 for this checkpoint) on an all-silent segment, and since
        # segments are combined with max(), a single silent segment pins the
        # whole file's score at that constant whenever the real speech scores
        # *below* it -- i.e. on exactly the genuine (non-spoofed) files.
        # Measured on the 100-file voice proxy set before this fix: 38/100
        # files (100% of them padded with >4s of silence by yusung's 10s-chunk
        # pipeline, all of them genuine) reported a bit-identical
        # 0.7642183899879456, unchanged even when the audio's loudness was
        # shifted by up to 9dB -- proof the value came from the silent
        # padding, not the speech. That inflated VOICE_FAKE_PROB for real
        # voices (hurting VoiceEER) and propagated into FILE_FAKE_PROB
        # (hurting FileEER, ADS weight 0.5). See outputs/voice_lufs_impact_predictions.csv
        # and scripts/evaluate_voice_lufs_impact.py for the diagnosis.
        segment_scores = []
        side_scores = []
        num_silent_segments = 0
        with torch.inference_mode():
            for start in _get_segment_starts(audio_np.size, self.cfg.segment_samples):
                segment = _extract_segment(audio_np, start, self.cfg.segment_samples)
                segment_rms = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
                if segment_rms < self.cfg.silence_rms_floor:
                    num_silent_segments += 1
                    continue
                segment_tensor = torch.from_numpy(segment).to(self._load_device)
                logits = self._model(input_values=segment_tensor)["logits"]
                probabilities = torch.softmax(logits.float(), dim=-1)
                segment_scores.append(float(probabilities[0, self._fake_label_index]))
                if self._side_conformer is not None:
                    side_logits, _ = self._side_conformer(self._side_in["x"])
                    side_scores.append(float(torch.softmax(side_logits.float(), dim=-1)[0, self._fake_label_index]))

        if not segment_scores:
            # Every segment was below the silence floor even though the
            # whole-file RMS was not -- treat as the silent case rather than
            # inventing a score from silence.
            elapsed = time.time() - t0
            logger.info(
                "DF-Arena: all %d segment(s) silent (file rms=%.2e), skipping inference",
                num_silent_segments, rms,
            )
            return {
                "fake_probability": 0.0,
                "real_probability": 1.0,
                "predicted_label": "real",
                "all_silent": True,
                "model": MODEL_LABEL,
                "timings": {"total": elapsed},
            }

        fake_prob = max(segment_scores)
        self.last_side_probability = max(side_scores) if side_scores else None
        elapsed = time.time() - t0
        logger.info(
            "DF-Arena: %d segments scored (%d silent, excluded), fake_probability=%.4f (%.3fs)",
            len(segment_scores), num_silent_segments, fake_prob, elapsed,
        )
        return {
            "fake_probability": fake_prob,
            "real_probability": 1.0 - fake_prob,
            "predicted_label": "fake" if fake_prob >= 0.5 else "real",
            "all_silent": False,
            "num_silent_segments": num_silent_segments,
            "num_scored_segments": len(segment_scores),
            "model": MODEL_LABEL,
            "timings": {"total": elapsed},
        }

    def predict_file(self, path: str | Path) -> Dict:
        waveform, sr = audio_io.load_audio(path, mono=False)
        return self.predict(waveform, sr)


# ---- component-isolation probe (2026-09-27) --------------------------------
def _load_side_conformer(model, device):
    """Fine-tuned copy of backbone.conformer, or None if not configured.

    The file holds only the parameters that were trained (last Conformer block
    + fc5); everything else, including the frozen BatchNorm running stats, is
    the published model's, so the copy starts from a deepcopy and overwrites
    exactly those keys. Any key mismatch is a hard error.
    """
    import copy
    path = os.environ.get("JAEHYUN_DFA_SIDE_CONFORMER", "")
    if not path:
        return None
    side = copy.deepcopy(model.backbone.conformer)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(side.named_parameters())
    unknown = sorted(set(blob) - set(params))
    if unknown:
        raise ValueError(f"side conformer: unknown keys {unknown[:5]}")
    with torch.no_grad():
        for k, v in blob.items():
            if tuple(v.shape) != tuple(params[k].shape):
                raise ValueError(f"side conformer: shape mismatch for {k}")
            params[k].copy_(v.to(params[k].dtype))
    logger.info("DF-Arena: side conformer loaded from %s (%d tensors overwritten)", path, len(blob))
    return side.to(device).eval()
