"""Top-level pipeline: combines AudioPresenceDetector (AudioMosaic) and
PolarFormerSeparator (BS PolarFormer) into a single Voice/Music presence
decision that uses separation as additional evidence rather than trusting
the original-mixture classifier alone, plus routes ArtifactNet (music
deepfake) and the WavLM voice-fake detector via music_routing's unified
zone-based processing plan.

See AUDIO_PRESENCE_SEPARATION_TASK.md sections 9-14 and
music_routing.py's module docstring for the routing scheme.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from . import audio_io, config
from .audio_presence import AudioPresenceDetector
from .artifactnet import ArtifactNetMusicDetector, MusicFakeCalibrator
from .music_fake_lofcz import LofczMusicFakeDetector
from .music_routing import decide_processing_plan
from .separator import PolarFormerSeparator
from .voice_fake_detector import VoiceFakeDetector

logger = logging.getLogger(__name__)


def energy_confidence(stem_rms: float, mixture_rms: float, energy_floor_db: float, eps: float = 1e-10) -> float:
    """Turn a stem's RMS energy into a [0, 1] confidence that the stem is
    "real" content and not just separator residual/artifact.

    Below `energy_floor_db` (absolute, dBFS) the stem is treated as silence
    (confidence 0) regardless of what the classifier says about it -- this is
    the "energy gating" required so that `stem_rms > 0` alone never implies
    presence (task doc section 11). Above the floor, confidence scales with
    the stem/mixture energy ratio, clipped to [0, 1].
    """
    stem_db = audio_io.rms_to_db(stem_rms, eps=eps)
    if stem_db < energy_floor_db:
        return 0.0
    ratio = stem_rms / (mixture_rms + eps)
    return float(min(1.0, max(0.0, ratio)))


def combine(
    original_score: float,
    stem_classifier_score: Optional[float],
    stem_energy_confidence: Optional[float],
    original_weight: float,
    stem_classifier_weight: float,
    stem_energy_weight: float,
) -> float:
    """Heuristic (NOT calibrated) weighted fusion of original/stem evidence.

    If separation was not run (both components' zones weren't LOW-vs-other,
    see music_routing.decide_processing_plan), stem_* args are None and we
    fall back to the original score alone.
    """
    if stem_classifier_score is None or stem_energy_confidence is None:
        return float(original_score)
    total_weight = original_weight + stem_classifier_weight + stem_energy_weight
    if total_weight <= 0:
        raise ValueError("Fusion weights must sum to a positive number")
    score = (
        original_weight * original_score
        + stem_classifier_weight * stem_classifier_score
        + stem_energy_weight * stem_energy_confidence
    ) / total_weight
    return float(min(1.0, max(0.0, score)))


class AudioPresencePipeline:
    def __init__(
        self,
        cfg: Optional[config.PipelineConfig] = None,
        device: Optional[torch.device] = None,
        hf_cache_dir: Path = config.DEFAULT_HF_CACHE_DIR,
    ) -> None:
        self.cfg = cfg or config.PipelineConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weight_sum = (
            self.cfg.presence.original_weight
            + self.cfg.presence.stem_classifier_weight
            + self.cfg.presence.stem_energy_weight
        )
        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(f"Fusion weights must sum to ~1.0, got {weight_sum}")

        # All models are loaded exactly once here and reused across every
        # file processed by this pipeline instance (task doc section 17).
        self.presence_detector = AudioPresenceDetector(self.cfg.presence, device=self.device, hf_cache_dir=hf_cache_dir)
        self.separator = PolarFormerSeparator(self.cfg.separator, device=self.device, hf_cache_dir=hf_cache_dir)

        self.music_detector = None
        if self.cfg.enable_artifactnet:
            if self.cfg.music_fake_backend == "lofcz":
                self.music_detector = LofczMusicFakeDetector(self.cfg.lofcz, device=self.device, hf_cache_dir=hf_cache_dir)
            else:
                self.music_detector = ArtifactNetMusicDetector(self.cfg.artifactnet, device=self.device, hf_cache_dir=hf_cache_dir)
        self.music_fake_calibrator = MusicFakeCalibrator()

        self.voice_fake_detector: Optional[VoiceFakeDetector] = None
        if self.cfg.enable_voice_fake_detection:
            self.voice_fake_detector = VoiceFakeDetector(self.cfg.voice_fake, device=self.device, hf_cache_dir=hf_cache_dir)

    # ------------------------------------------------------------------
    def analyze(
        self,
        waveform,
        sample_rate: int,
        file_name: str = "audio",
        output_dir: Optional[str | Path] = None,
        force_separation: bool = False,
    ) -> Dict:
        timings: Dict[str, float] = {}
        t_total0 = time.time()

        if isinstance(waveform, np.ndarray):
            mixture_np = waveform if waveform.ndim == 1 else waveform.mean(axis=0)
        else:
            mixture_np = waveform.numpy() if waveform.dim() == 1 else waveform.mean(dim=0).numpy()

        t0 = time.time()
        original = self.presence_detector.predict(waveform, sample_rate)
        timings["audiomosaic_original"] = time.time() - t0

        # --- unified processing plan (music_routing.py) -------------------
        # Decided from the *original* AudioMosaic scores, before any
        # separation happens. Single source of truth for: whether BS
        # PolarFormer runs, and what input (if any) each fake-detector uses.
        plan = decide_processing_plan(original["voice_probability"], original["music_probability"], self.cfg.routing)
        do_separate = plan.run_separation or force_separation

        # Ablation overrides (config.PipelineConfig.force_artifactnet_original /
        # force_voice_fake_original) -- see their docstring. Only changes
        # which audio a detector scores; `do_separate` above (and therefore
        # whether stems exist at all for presence-fusion CPS) is untouched.
        artifactnet_input = "original" if self.cfg.force_artifactnet_original else plan.artifactnet_input
        voice_detector_input = "original" if self.cfg.force_voice_fake_original else plan.voice_detector_input

        stems_paths = {"voice": None, "music": None}
        vocal_stem_voice_score = None
        instrumental_stem_music_score = None
        vocal_energy_conf = None
        instrumental_energy_conf = None
        sep_diag = {}
        sep_result = None

        if do_separate:
            t0 = time.time()
            sep_result = self.separator.separate(waveform, sample_rate)
            timings["polarformer_separation"] = time.time() - t0
            sep_diag = sep_result["diagnostics"]

            if output_dir is not None:
                stems_paths = self._write_stems(sep_result, output_dir, file_name)

            t0 = time.time()
            vocal_probs = self.presence_detector.predict(sep_result["vocals"], sample_rate)
            instrumental_probs = self.presence_detector.predict(sep_result["instrumental"], sample_rate)
            timings["audiomosaic_stems"] = time.time() - t0

            vocal_stem_voice_score = vocal_probs["voice_probability"]
            instrumental_stem_music_score = instrumental_probs["music_probability"]

            vocal_energy_conf = energy_confidence(
                sep_diag["vocal_rms"], sep_diag["mixture_rms"], self.cfg.presence.energy_floor_db
            )
            instrumental_energy_conf = energy_confidence(
                sep_diag["instrumental_rms"], sep_diag["mixture_rms"], self.cfg.presence.energy_floor_db
            )
        else:
            timings["polarformer_separation"] = 0.0
            timings["audiomosaic_stems"] = 0.0
            logger.info(
                "Plan '%s' for '%s': skipping BS PolarFormer (music_zone=%s, voice_zone=%s)",
                plan.label, file_name, plan.music_zone, plan.voice_zone,
            )

        # --- ArtifactNet (music fake) --------------------------------------
        artifactnet_diag: Dict = {"used": False, "used_for_final": False}
        music_fake_applicable = False
        music_fake_probability = None
        if self.music_detector is not None:
            if artifactnet_input == "original":
                t0 = time.time()
                art_result = self.music_detector.predict(waveform, sample_rate)
                timings["artifactnet"] = art_result["timings"]["total"]
                music_fake_applicable = True
                raw_prob = art_result["fake_probability"]
                music_fake_probability = self.music_fake_calibrator.transform(raw_prob)
                artifactnet_diag = {
                    "used": True,
                    "used_for_final": True,
                    "model": art_result["model"],
                    "version": art_result["model_version"],
                    "input_type": "original",
                    "fake_probability": art_result["fake_probability"],
                    "real_probability": art_result["real_probability"],
                    "artifactnet_raw_probability": raw_prob,
                    "aggregation": art_result["aggregation"],
                    "num_segments": art_result["num_segments"],
                    "num_valid_segments": art_result["num_valid_segments"],
                    "segment_probabilities": art_result["segment_probabilities"],
                    "segments": art_result["segments"],
                    "stats": art_result["stats"],
                    "predicted_label": art_result["predicted_label"],
                }
            elif artifactnet_input == "separated_instrumental":
                assert sep_result is not None
                t0 = time.time()
                # Use the pre-downsample native-rate (44.1kHz) stem directly,
                # not sep_result["instrumental"] (already resampled back down
                # to `sample_rate` by the separator). ArtifactNet itself
                # always resamples its input to 44.1kHz, so feeding it the
                # original-rate stem would force a wasted
                # sample_rate -> 44100 -> sample_rate -> 44100 round trip
                # whenever sample_rate != 44100 (the common case: DACON audio
                # is 16kHz) -- see separator.py's return-value docstring.
                art_result = self.music_detector.predict(
                    sep_result["instrumental_native"], sep_result["native_sample_rate"]
                )
                timings["artifactnet"] = art_result["timings"]["total"]
                music_fake_applicable = True
                raw_prob = art_result["fake_probability"]
                music_fake_probability = self.music_fake_calibrator.transform(raw_prob)
                artifactnet_diag = {
                    "used": True,
                    "used_for_final": True,
                    "model": art_result["model"],
                    "version": art_result["model_version"],
                    "input_type": "separated_instrumental",
                    "fake_probability": art_result["fake_probability"],
                    "real_probability": art_result["real_probability"],
                    "artifactnet_raw_probability": raw_prob,
                    "aggregation": art_result["aggregation"],
                    "num_segments": art_result["num_segments"],
                    "num_valid_segments": art_result["num_valid_segments"],
                    "segment_probabilities": art_result["segment_probabilities"],
                    "segments": art_result["segments"],
                    "stats": art_result["stats"],
                    "predicted_label": art_result["predicted_label"],
                }
        else:
            timings["artifactnet"] = 0.0

        # --- WavLM (voice fake) --------------------------------------------
        voice_fake_probability = None
        voice_fake_diag: Dict = {"used": False}
        if self.voice_fake_detector is not None:
            if voice_detector_input == "original":
                t0 = time.time()
                vf_result = self.voice_fake_detector.predict(waveform, sample_rate)
                timings["voice_fake_detection"] = vf_result["timings"]["total"]
                voice_fake_probability = vf_result["fake_probability"]
                voice_fake_diag = {
                    "used": True,
                    "input_type": "original",
                    "model": vf_result["model"],
                    "fake_probability": vf_result["fake_probability"],
                    "real_probability": vf_result["real_probability"],
                    "predicted_label": vf_result["predicted_label"],
                    "all_silent": vf_result["all_silent"],
                }
            elif voice_detector_input == "separated_vocal":
                assert sep_result is not None
                t0 = time.time()
                vf_result = self.voice_fake_detector.predict(sep_result["vocals"], sample_rate)
                timings["voice_fake_detection"] = vf_result["timings"]["total"]
                voice_fake_probability = vf_result["fake_probability"]
                voice_fake_diag = {
                    "used": True,
                    "input_type": "separated_vocal",
                    "model": vf_result["model"],
                    "fake_probability": vf_result["fake_probability"],
                    "real_probability": vf_result["real_probability"],
                    "predicted_label": vf_result["predicted_label"],
                    "all_silent": vf_result["all_silent"],
                }
        else:
            timings["voice_fake_detection"] = 0.0

        voice_final = combine(
            original["voice_probability"],
            vocal_stem_voice_score,
            vocal_energy_conf,
            self.cfg.presence.original_weight,
            self.cfg.presence.stem_classifier_weight,
            self.cfg.presence.stem_energy_weight,
        )
        music_final = combine(
            original["music_probability"],
            instrumental_stem_music_score,
            instrumental_energy_conf,
            self.cfg.presence.original_weight,
            self.cfg.presence.stem_classifier_weight,
            self.cfg.presence.stem_energy_weight,
        )

        voice_present = voice_final >= self.cfg.presence.voice_threshold
        music_present = music_final >= self.cfg.presence.music_threshold

        timings["total"] = time.time() - t_total0

        gpu_peak_mb = None
        if self.device.type == "cuda":
            gpu_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        diagnostics = {
            "mixture_rms": sep_diag.get("mixture_rms", audio_io.compute_rms(mixture_np)),
            "vocal_rms": sep_diag.get("vocal_rms"),
            "instrumental_rms": sep_diag.get("instrumental_rms"),
            "vocal_energy_ratio": sep_diag.get("vocal_energy_ratio"),
            "instrumental_energy_ratio": sep_diag.get("instrumental_energy_ratio"),
            "reconstruction_error_rms": sep_diag.get("reconstruction_error_rms"),
            "original_voice_score": original["voice_probability"],
            "original_music_score": original["music_probability"],
            "vocal_stem_voice_score": vocal_stem_voice_score,
            "instrumental_stem_music_score": instrumental_stem_music_score,
            "vocal_energy_confidence": vocal_energy_conf,
            "instrumental_energy_confidence": instrumental_energy_conf,
            "separated": do_separate,
            "music_zone": plan.music_zone,
            "voice_zone": plan.voice_zone,
            "fusion_weights": {
                "original_weight": self.cfg.presence.original_weight,
                "stem_classifier_weight": self.cfg.presence.stem_classifier_weight,
                "stem_energy_weight": self.cfg.presence.stem_energy_weight,
            },
            "thresholds": {
                "voice_threshold": self.cfg.presence.voice_threshold,
                "music_threshold": self.cfg.presence.music_threshold,
            },
            "routing_thresholds": {
                "low_threshold": self.cfg.routing.low_threshold,
                "high_threshold": self.cfg.routing.high_threshold,
            },
            "timings": timings,
            "gpu_peak_memory_mb": gpu_peak_mb,
        }

        logger.info(
            "Pipeline result '%s': voice=%.4f music=%.4f voice_present=%s music_present=%s "
            "(separated=%s, route=%s, music_fake=%s, voice_fake=%s, total=%.3fs)",
            file_name,
            voice_final,
            music_final,
            voice_present,
            music_present,
            do_separate,
            plan.label,
            music_fake_probability,
            voice_fake_probability,
            timings["total"],
        )

        return {
            "file": file_name,
            "duration": len(mixture_np) / sample_rate,
            "sample_rate": sample_rate,
            "voice_probability": voice_final,
            "music_probability": music_final,
            "voice_present": bool(voice_present),
            "music_present": bool(music_present),
            "stems": stems_paths,
            "route": plan.label,
            "music_fake_applicable": music_fake_applicable,
            "music_fake_probability": music_fake_probability,
            "artifactnet": artifactnet_diag,
            "voice_fake_probability": voice_fake_probability,
            "voice_fake": voice_fake_diag,
            "diagnostics": diagnostics,
            "windows": original["windows"],
        }

    def _write_stems(self, sep_result: Dict, output_dir: str | Path, file_name: str) -> Dict:
        import soundfile as sf

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file_name).stem

        voice_path = output_dir / f"{stem}_voice.wav"
        music_path = output_dir / f"{stem}_music.wav"

        def _write(p: Path, wav: np.ndarray) -> None:
            data = wav.T if wav.ndim == 2 else wav
            sf.write(str(p), data, sep_result["sample_rate"])

        _write(voice_path, sep_result["vocals"])
        _write(music_path, sep_result["instrumental"])
        return {"voice": str(voice_path), "music": str(music_path)}

    def analyze_file(self, path: str | Path, output_dir: Optional[str | Path] = None, force_separation: bool = False) -> Dict:
        t0 = time.time()
        waveform, sr = audio_io.load_audio(path, mono=False)
        load_time = time.time() - t0
        result = self.analyze(
            waveform, sr, file_name=Path(path).name, output_dir=output_dir, force_separation=force_separation
        )
        result["diagnostics"]["timings"]["audio_load"] = load_time
        return result
