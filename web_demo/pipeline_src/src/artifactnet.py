"""ArtifactNetMusicDetector: AI-generated-music detection via the official
`intrect/artifactnet` v9.4 end-to-end ONNX build (STFT -> UNet -> HPSS ->
7-channel CNN -> sigmoid, all inside the ONNX graph). We only handle audio
loading, mono conversion, 44.1kHz resampling, 4-second segmentation, and
segment -> song-level aggregation -- never STFT/mel/HPSS/normalization,
which are already baked into the ONNX graph.

License: CC BY-NC 4.0 (non-commercial only) -- see README.md "License".

See ARTIFACTNET_MUSIC_DETECTOR_TASK.md.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from . import audio_io, config, onnx_utils

logger = logging.getLogger(__name__)


class MusicFakeCalibrator:
    """Identity transform for now. No DACON validation set exists yet to fit
    a real calibrator against (task doc section 23). Swap `transform` for a
    fitted Platt scaling / isotonic regression / temperature scaling once one
    does; keep `raw_probability` around wherever this is used so recalibrating
    later doesn't lose the original ArtifactNet score.
    """

    def transform(self, score: float) -> float:
        return float(score)


class ArtifactNetLoadError(RuntimeError):
    pass


class ArtifactNetInferenceError(RuntimeError):
    pass


# Both HPSS soft-mask Div nodes in the official graph divide by this single
# Add node's output (harmonic_energy^p + percussive_energy^p). Verified via
# graph tracing + intermediate-tensor inspection on a real reproducible NaN
# case (both numerator and denominator exactly 0.0 at the NaN location).
_HPSS_DIV_DENOMINATOR = "add_35"
_HPSS_DIV_EPS = 1e-20  # far below float32 precision unless denominator == 0 exactly


def _build_nan_safety_patched_onnx(onnx_path: Path) -> Path:
    """Return a cached sibling of `onnx_path` with an epsilon-guarded HPSS
    mask denominator (see ArtifactNetConfig.apply_nan_safety_patch). Written
    once into the same directory as the original files (external weight data
    stays shared via the unmodified .onnx.data, never duplicated); reused on
    subsequent loads.
    """
    patched_path = onnx_path.with_name(onnx_path.stem + "_nansafe.onnx")
    if patched_path.is_file():
        return patched_path

    import onnx
    from onnx import helper, numpy_helper

    logger.info("Building NaN-safety-patched ArtifactNet graph (cached at %s)...", patched_path)
    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph

    producer_idx = next((i for i, n in enumerate(graph.node) if _HPSS_DIV_DENOMINATOR in n.output), None)
    if producer_idx is None:
        raise ArtifactNetLoadError(
            f"NaN-safety patch: could not find a node producing '{_HPSS_DIV_DENOMINATOR}' in the graph "
            "(model structure may have changed since this patch was written). Set "
            "ArtifactNetConfig(apply_nan_safety_patch=False) to use the official graph unmodified."
        )

    eps_name = "hpss_div_eps"
    safe_name = f"{_HPSS_DIV_DENOMINATOR}_safe"
    graph.initializer.append(numpy_helper.from_array(np.array(_HPSS_DIV_EPS, dtype=np.float32), name=eps_name))
    graph.node.insert(
        producer_idx + 1,
        helper.make_node("Add", inputs=[_HPSS_DIV_DENOMINATOR, eps_name], outputs=[safe_name], name="node_hpss_div_eps_patch"),
    )

    patched_count = 0
    for n in graph.node:
        if n.op_type == "Div" and len(n.input) == 2 and n.input[1] == _HPSS_DIV_DENOMINATOR:
            n.input[1] = safe_name
            patched_count += 1
    if patched_count != 2:
        raise ArtifactNetLoadError(
            f"NaN-safety patch: expected exactly 2 Div nodes on '{_HPSS_DIV_DENOMINATOR}', found {patched_count}. "
            "Refusing to apply an unverified patch -- model graph structure may have changed. Set "
            "ArtifactNetConfig(apply_nan_safety_patch=False) to use the official graph unmodified."
        )

    onnx.save(model, str(patched_path))
    logger.info("NaN-safety patch applied (%d Div node(s) now epsilon-guarded); original files untouched.", patched_count)
    return patched_path


class ArtifactNetMusicDetector:
    """AI-generated-music probability via ArtifactNet v9.4 (ONNX)."""

    def __init__(
        self,
        cfg: Optional[config.ArtifactNetConfig] = None,
        device: Optional[torch.device] = None,
        hf_cache_dir: Path = config.DEFAULT_HF_CACHE_DIR,
    ) -> None:
        self.cfg = cfg or config.ArtifactNetConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.ensure_hf_cache_env(hf_cache_dir)

        t0 = time.time()
        from huggingface_hub import snapshot_download

        logger.info("Loading ArtifactNet model '%s' (device=%s)...", self.cfg.model_id, self.device)
        repo_dir = Path(
            snapshot_download(self.cfg.model_id, allow_patterns=["*.onnx", "*.onnx.data"])
        )

        onnx_path = repo_dir / "artifactnet_v94_full.onnx"
        data_path = repo_dir / "artifactnet_v94_full.onnx.data"
        missing = [p.name for p in (onnx_path, data_path) if not p.is_file()]
        if missing:
            raise ArtifactNetLoadError(
                f"ArtifactNet download incomplete in {repo_dir}: missing {missing}. "
                "Both artifactnet_v94_full.onnx and artifactnet_v94_full.onnx.data "
                "must be present in the same directory (external-data ONNX format)."
            )

        if self.cfg.apply_nan_safety_patch:
            session_onnx_path = _build_nan_safety_patched_onnx(onnx_path)
        else:
            session_onnx_path = onnx_path
            logger.warning(
                "ArtifactNet NaN-safety patch disabled: expect NaN P(AI) for a large fraction of segments "
                "(measured ~44%% CPU / ~84%% GPU on real audio) -- see README ArtifactNet section."
            )

        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        requested = onnx_utils.pick_onnx_providers(prefer_gpu=self.cfg.prefer_gpu and self.device.type == "cuda")
        logger.info("ArtifactNet ONNX providers:\n- %s", "\n- ".join(available_providers))
        # `.onnx` file alone is enough for ort to find the sibling `.onnx.data`
        # (same directory, standard external-data convention) -- do not move
        # or copy just the .onnx file elsewhere.
        self.session = ort.InferenceSession(str(session_onnx_path), providers=requested)
        self.using_gpu = onnx_utils.resolve_active_provider(self.session, requested, "ArtifactNetMusicDetector")
        logger.info("Selected:\n%s", self.session.get_providers()[0])

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ArtifactNetLoadError(
                f"Expected exactly 1 input and 1 output, got inputs={[i.name for i in inputs]} "
                f"outputs={[o.name for o in outputs]}"
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        logger.info("ArtifactNet input: name=%r shape=%s dtype=%s", self.input_name, inputs[0].shape, inputs[0].type)
        logger.info("ArtifactNet output: name=%r shape=%s dtype=%s", self.output_name, outputs[0].shape, outputs[0].type)

        self.segment_samples = int(round(self.cfg.sample_rate * self.cfg.segment_seconds))
        static_dims = [d for d in inputs[0].shape if isinstance(d, int)]
        if static_dims and static_dims[-1] != self.segment_samples:
            raise ArtifactNetLoadError(
                f"ArtifactNet input's last static dim is {static_dims[-1]}, expected "
                f"{self.segment_samples} (= {self.cfg.sample_rate}Hz x {self.cfg.segment_seconds}s). "
                "The official spec (44.1kHz, 4s mono) may have changed -- do not silently override."
            )

        self.supports_batch = self._probe_batch_support()
        logger.info(
            "ArtifactNet loaded in %.2fs (segment_samples=%d, batch_inference=%s)",
            time.time() - t0,
            self.segment_samples,
            self.supports_batch,
        )

    def _probe_batch_support(self) -> bool:
        """The graph declares a dynamic 'batch' input dim, but that does not
        guarantee the graph actually supports batch>1 (internal reshape/Gemm
        nodes can hardcode batch=1). Verified once at load time with a real
        batch=2 call; never assumed. See task doc section 25."""
        try:
            dummy = np.zeros((2, self.segment_samples), dtype=np.float32)
            out = self.session.run([self.output_name], {self.input_name: dummy})[0]
            return out.shape[0] == 2
        except Exception as exc:  # noqa: BLE001 - any failure means "no batch support"
            logger.info("ArtifactNet batch>1 inference not supported (%s); using sequential (batch=1) calls.", exc)
            return False

    # ------------------------------------------------------------------
    def _make_segments(self, mono_waveform: torch.Tensor, num_samples: int) -> List[Dict]:
        window = self.segment_samples
        hop = max(1, int(round(self.cfg.hop_seconds * self.cfg.sample_rate)))
        segments: List[Dict] = []
        start = 0
        while start < num_samples:
            end = min(start + window, num_samples)
            chunk = mono_waveform[start:end]
            padded = False
            if chunk.shape[-1] < window:
                # Never drop the only segment of a sub-4s file (task doc
                # section 11); "skip" only applies to a *tail* segment after
                # at least one full segment already exists.
                if self.cfg.tail_policy == "skip" and start != 0:
                    break
                chunk = torch.nn.functional.pad(chunk, (0, window - chunk.shape[-1]))
                padded = True
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "data": chunk,
                    "valid_duration": (end - start) / self.cfg.sample_rate,
                    "padded": padded,
                }
            )
            if end >= num_samples:
                break
            start += hop
        return segments

    def _aggregate(self, probs: np.ndarray) -> float:
        method = self.cfg.aggregation
        if method == "median":
            return float(np.median(probs))
        if method == "mean":
            return float(np.mean(probs))
        if method == "max":
            return float(np.max(probs))
        if method == "topk_mean":
            k = min(probs.size, max(1, int(round(probs.size * self.cfg.top_k_ratio))))
            return float(np.partition(probs, -k)[-k:].mean())
        raise ValueError(f"Unknown aggregation method: {method}")

    def _run_segments(self, segments: List[Dict]) -> List[float]:
        if self.supports_batch and len(segments) > 1:
            batch = np.stack([seg["data"].numpy() for seg in segments]).astype(np.float32)
            out = self.session.run([self.output_name], {self.input_name: batch})[0]
            return [float(v) for v in out.reshape(-1)]
        probs = []
        for seg in segments:
            x = seg["data"].numpy().reshape(1, -1).astype(np.float32)
            out = self.session.run([self.output_name], {self.input_name: x})[0]
            probs.append(float(out.reshape(-1)[0]))
        return probs

    # ------------------------------------------------------------------
    def predict(self, waveform, sample_rate: int) -> Dict:
        """Run ArtifactNet on a waveform.

        Args:
            waveform: numpy array or torch tensor. Explicit shape check (no
                axis guessing): 1D [T] mono, or 2D [C, T] channels-first
                (torchaudio convention, matches audio_io.load_audio).
            sample_rate: sample rate of `waveform`.
        """
        t0 = time.time()
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        waveform = waveform.float()

        if waveform.dim() == 1:
            channels_first = waveform.unsqueeze(0)  # [1, T]
        elif waveform.dim() == 2:
            channels_first = waveform  # assumed [C, T], matching audio_io convention
        else:
            raise ValueError(f"ArtifactNet expects a 1D [T] or 2D [C, T] waveform, got shape {tuple(waveform.shape)}")

        if channels_first.shape[-1] == 0:
            raise ValueError("Cannot run ArtifactNet on an empty waveform")

        mono = audio_io.to_mono(channels_first)  # [1, T]

        t_resample0 = time.time()
        resampled = audio_io.resample_torch(mono, sample_rate, self.cfg.sample_rate)[0]  # [T] @ 44100
        resample_time = time.time() - t_resample0

        resampled_np = resampled.numpy()
        audio_io.validate_finite(resampled_np, "ArtifactNet input audio")

        num_samples = resampled.shape[-1]
        duration = num_samples / self.cfg.sample_rate

        segments = self._make_segments(resampled, num_samples)

        t_infer0 = time.time()
        seg_probs = self._run_segments(segments)
        infer_time = time.time() - t_infer0

        # The released ONNX build occasionally produces NaN/Inf for specific
        # real-world segments -- reproducible, deterministic, and unrelated to
        # our preprocessing (input waveform verified finite/normal-range
        # beforehand; same NaN reproduces on a brand-new session with no
        # other calls made). This is a model-internal numerical edge case,
        # not something we can or should "fix" by touching the ONNX graph.
        # We exclude non-finite segments from aggregation rather than let a
        # single NaN segment silently poison the whole file's score.
        all_probs = np.array(seg_probs, dtype=np.float64)
        valid_mask = np.isfinite(all_probs)
        num_invalid = int((~valid_mask).sum())
        if num_invalid:
            logger.warning(
                "ArtifactNet returned non-finite P(AI) for %d/%d segment(s) (model-internal numerical "
                "edge case, not a preprocessing bug -- see artifactnet.py docstring); excluding them "
                "from aggregation.",
                num_invalid,
                len(seg_probs),
            )
        probs_arr = all_probs[valid_mask]
        if probs_arr.size == 0:
            raise ArtifactNetInferenceError(
                f"All {len(seg_probs)} segment(s) returned non-finite P(AI); cannot compute a song-level score."
            )

        fake_probability = self._aggregate(probs_arr)
        real_probability = 1.0 - fake_probability
        predicted_label = "ai_generated" if fake_probability > self.cfg.predicted_label_threshold else "human_real"

        segments_out = [
            {
                "start": seg["start"] / self.cfg.sample_rate,
                "end": seg["end"] / self.cfg.sample_rate,
                "valid_duration": seg["valid_duration"],
                "padded": seg["padded"],
                "fake_probability": prob,
                "valid": bool(np.isfinite(prob)),
            }
            for seg, prob in zip(segments, seg_probs)
        ]

        total_time = time.time() - t0
        logger.info(
            "ArtifactNet: %d segments (%d valid), P(AI)=%.4f (%s) in %.3fs",
            len(segments),
            probs_arr.size,
            fake_probability,
            self.cfg.aggregation,
            total_time,
        )

        return {
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "predicted_label": predicted_label,
            "segment_probabilities": seg_probs,
            "aggregation": self.cfg.aggregation,
            "num_segments": len(segments),
            "num_valid_segments": int(probs_arr.size),
            "num_invalid_segments": num_invalid,
            "input_duration": duration,
            "model": self.cfg.model_id,
            "model_version": "v9.4",
            "segments": segments_out,
            "stats": {
                "median": float(np.median(probs_arr)),
                "mean": float(np.mean(probs_arr)),
                "std": float(np.std(probs_arr)),
                "min": float(np.min(probs_arr)),
                "max": float(np.max(probs_arr)),
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
