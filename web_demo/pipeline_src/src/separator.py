"""PolarFormerSeparator: Voice/Music source separation via BS PolarFormer
(ONNX Runtime), reusing the *official* `run_onnx_inference.py` STFT / chunk /
overlap / mask / iSTFT implementation vendored inside the bgkb/bs_polarformer
HF repo (functions `prepare_stft` / `reconstruct_audio` are imported and
called as-is; only the I/O, resampling and chunk-size-by-device policy layer
around them is ours).

See AUDIO_PRESENCE_SEPARATION_TASK.md sections 7-9, 21.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from . import audio_io, config, onnx_utils

logger = logging.getLogger(__name__)


class PolarFormerSeparator:
    """Voice/Music separation via BS PolarFormer (44.1kHz stereo pretrained domain)."""

    # Empirically measured on this machine: the official 20s chunk_size
    # (882000 samples @44.1kHz) caused CPU-side ONNX Runtime memory use to
    # exceed 40GB+ RSS on a 62GB shared host (likely dense-attention fallback
    # for the exported graph, since flash_attn kernels have no ONNX CPU
    # equivalent). A 5s chunk (220500 samples) completed the same audio with
    # a stable ~8GB RSS. GPU inference does not hit this and uses the
    # official default. This is a runtime safety measure, not a change to
    # the separation algorithm itself (same STFT/mask/iSTFT code either way).
    CPU_SAFE_CHUNK_SIZE = 220500

    def __init__(
        self,
        cfg: Optional[config.SeparatorConfig] = None,
        device: Optional[torch.device] = None,
        hf_cache_dir: Path = config.DEFAULT_HF_CACHE_DIR,
    ) -> None:
        self.cfg = cfg or config.SeparatorConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.ensure_hf_cache_env(hf_cache_dir)

        t0 = time.time()
        from huggingface_hub import snapshot_download

        logger.info("Loading BS PolarFormer model '%s' (device=%s)...", self.cfg.model_id, self.device)
        repo_dir = snapshot_download(
            self.cfg.model_id,
            allow_patterns=["*.onnx", "*.yaml", "run_onnx_inference.py"],
        )
        self._repo_dir = repo_dir
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        # Reuse the official, numerically-validated STFT/mask/iSTFT functions.
        from run_onnx_inference import prepare_stft, reconstruct_audio  # type: ignore

        self._prepare_stft = prepare_stft
        self._reconstruct_audio = reconstruct_audio

        import yaml

        model_cfg_path = Path(repo_dir) / "model_bs_polarformer_float16.yaml"
        with open(model_cfg_path) as f:
            self.model_cfg = yaml.full_load(f)

        self.stft_kwargs = dict(
            n_fft=self.model_cfg["model"]["stft_n_fft"],
            hop_length=self.model_cfg["model"]["stft_hop_length"],
            win_length=self.model_cfg["model"]["stft_win_length"],
            normalized=self.model_cfg["model"].get("stft_normalized", False),
        )
        self.audio_channels = 2 if self.model_cfg["model"].get("stereo", False) else 1
        assert self.audio_channels == 2, "This implementation assumes the stereo BS PolarFormer config."

        onnx_filename = "bs_polarformer_fp16.onnx" if self.cfg.use_fp16 else "bs_polarformer.onnx"
        onnx_path = Path(repo_dir) / onnx_filename
        import onnxruntime as ort

        providers = onnx_utils.pick_onnx_providers(prefer_gpu=self.cfg.prefer_gpu and self.device.type == "cuda")
        logger.info("ONNX Runtime providers: %s (model=%s)", providers, onnx_filename)
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        # Check what onnxruntime actually loaded, not what we requested (see
        # onnx_utils.resolve_active_provider docstring): trusting the requested
        # list here previously caused a real bug -- it kept using the large
        # official GPU chunk_size while actually running on CPU, which is
        # exactly the config that caused the CPU memory blowup (see class
        # docstring below).
        self.using_gpu = onnx_utils.resolve_active_provider(self.session, providers, "PolarFormerSeparator")

        if self.cfg.chunk_size is not None:
            self.chunk_size = self.cfg.chunk_size
        elif self.using_gpu:
            self.chunk_size = self.model_cfg["inference"]["chunk_size"]  # official default (882000)
        else:
            self.chunk_size = self.CPU_SAFE_CHUNK_SIZE
            logger.info(
                "CPU inference: using chunk_size=%d (safety-reduced from official %d; see class docstring).",
                self.chunk_size,
                self.model_cfg["inference"]["chunk_size"],
            )
        self.num_overlap = self.cfg.num_overlap
        self.onnx_batch_size = self.cfg.onnx_batch_size

        logger.info("BS PolarFormer loaded in %.2fs (chunk_size=%d, num_overlap=%d)", time.time() - t0, self.chunk_size, self.num_overlap)

    # ------------------------------------------------------------------
    def _run_core_chunked(self, stereo_44k: torch.Tensor) -> torch.Tensor:
        """stereo_44k: [2, T] float32 torch tensor at 44100Hz. Returns vocals [2, T]."""
        audio_np = stereo_44k.numpy().astype(np.float32)
        total_samples = audio_np.shape[1]
        chunk_size = self.chunk_size
        step = max(1, chunk_size // self.num_overlap)

        result = np.zeros((2, total_samples), dtype=np.float32)
        count = np.zeros(total_samples, dtype=np.float32)

        starts = list(range(0, total_samples, step))
        for start in starts:
            end = min(start + chunk_size, total_samples)
            chunk = audio_np[:, start:end]
            if chunk.shape[1] < chunk_size:
                pad = np.zeros((2, chunk_size - chunk.shape[1]), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=1)

            x, stft_repr, stft_window, raw_audio = self._prepare_stft(chunk, self.stft_kwargs, self.stft_kwargs["win_length"])
            mask = self.session.run(None, {"stft_features": x.numpy()})[0]
            recon = self._reconstruct_audio(
                stft_repr, mask, self.stft_kwargs, stft_window, self.audio_channels, raw_audio.shape[-1], zero_dc=True
            )
            recon = recon[0, 0].numpy()  # [2, chunk_size]

            actual_len = end - start
            result[:, start:end] += recon[:, :actual_len]
            count[start:end] += 1.0

        count = np.maximum(count, 1.0)
        result = result / count[np.newaxis, :]
        return torch.from_numpy(result)

    def separate(self, waveform, sample_rate: int) -> Dict:
        """Separate voice (vocals) and music (instrumental) stems.

        Args:
            waveform: numpy array or torch tensor, shape [T] (mono) or [C, T].
            sample_rate: sample rate of `waveform`.

        Returns:
            {"vocals": np.ndarray, "instrumental": np.ndarray, "sample_rate": int,
             "vocals_native": np.ndarray, "instrumental_native": np.ndarray,
             "native_sample_rate": int, "diagnostics": {...}}
            "vocals"/"instrumental" match the input's shape convention (mono
            [T] or stereo [C, T]) and sample rate. "*_native" are the
            pre-downsample stems still at this model's own processing domain
            (stereo [2, T] @ native_sample_rate, i.e. cfg.target_sample_rate)
            -- expose these so a caller needing 44.1kHz (e.g. ArtifactNet,
            which is 44.1kHz-native) can use them directly instead of paying
            for an input_sr -> 44100 -> input_sr -> 44100 round trip when
            input_sr != 44100 (this happened in practice for DACON's 16kHz
            audio before this was added; see README's routing/kHz section).
        """
        t0 = time.time()
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        waveform = waveform.float()
        orig_was_mono = waveform.dim() == 1 or waveform.shape[0] == 1
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        orig_num_channels = waveform.shape[0]
        orig_num_samples = waveform.shape[-1]

        # original -> safe float -> resample to 44100 -> stereo duplicate if mono
        target_sr = self.cfg.target_sample_rate
        resampled = audio_io.resample_torch(waveform, sample_rate, target_sr)
        stereo = audio_io.to_stereo(resampled)

        vocals_44k = self._run_core_chunked(stereo)
        # instrumental = mixture - vocals (mixture-consistency; also matches
        # the official run_onnx_inference.py's own "other" computation).
        instrumental_44k = stereo[:, : vocals_44k.shape[-1]] - vocals_44k

        # -> original sample rate -> original channel policy -> exact original length
        vocals_back = audio_io.resample_torch(vocals_44k, target_sr, sample_rate)
        instrumental_back = audio_io.resample_torch(instrumental_44k, target_sr, sample_rate)

        if orig_was_mono:
            vocals_back = audio_io.to_mono(vocals_back)
            instrumental_back = audio_io.to_mono(instrumental_back)
        else:
            vocals_back = vocals_back[:orig_num_channels]
            instrumental_back = instrumental_back[:orig_num_channels]

        vocals_back = audio_io.fit_length(vocals_back, orig_num_samples)
        instrumental_back = audio_io.fit_length(instrumental_back, orig_num_samples)

        if orig_was_mono:
            vocals_out = vocals_back[0]
            instrumental_out = instrumental_back[0]
        else:
            vocals_out = vocals_back
            instrumental_out = instrumental_back

        vocals_np = vocals_out.numpy()
        instrumental_np = instrumental_out.numpy()
        mixture_np = (waveform[0] if orig_was_mono else waveform).numpy()

        audio_io.validate_finite(vocals_np, "separated vocals")
        audio_io.validate_finite(instrumental_np, "separated instrumental")

        mixture_rms = audio_io.compute_rms(mixture_np)
        vocal_rms = audio_io.compute_rms(vocals_np)
        instrumental_rms = audio_io.compute_rms(instrumental_np)
        eps = 1e-10
        reconstruction = vocals_np + instrumental_np
        reconstruction_error = audio_io.compute_rms(reconstruction - mixture_np)

        diagnostics = {
            "mixture_rms": mixture_rms,
            "vocal_rms": vocal_rms,
            "instrumental_rms": instrumental_rms,
            "vocal_energy_ratio": vocal_rms / (mixture_rms + eps),
            "instrumental_energy_ratio": instrumental_rms / (mixture_rms + eps),
            "vocal_clipped": audio_io.detect_clipping(vocals_np),
            "instrumental_clipped": audio_io.detect_clipping(instrumental_np),
            "reconstruction_error_rms": reconstruction_error,
            "inference_time": time.time() - t0,
            "device": "cuda" if self.using_gpu else "cpu",
        }
        logger.info(
            "PolarFormer separation done in %.3fs (vocal_rms=%.5f instrumental_rms=%.5f recon_err=%.6f)",
            diagnostics["inference_time"],
            vocal_rms,
            instrumental_rms,
            reconstruction_error,
        )

        assert vocals_np.shape[-1] == orig_num_samples
        assert instrumental_np.shape[-1] == orig_num_samples

        return {
            "vocals": vocals_np,
            "instrumental": instrumental_np,
            "sample_rate": sample_rate,
            "vocals_native": vocals_44k.numpy(),
            "instrumental_native": instrumental_44k.numpy(),
            "native_sample_rate": target_sr,
            "diagnostics": diagnostics,
        }

    def separate_file(self, path: str | Path, output_dir: str | Path) -> Dict:
        waveform, sr = audio_io.load_audio(path, mono=False)
        result = self.separate(waveform, sr)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem

        import soundfile as sf

        voice_path = output_dir / f"{stem}_voice.wav"
        music_path = output_dir / f"{stem}_music.wav"

        def _write(p: Path, wav: np.ndarray) -> None:
            data = wav.T if wav.ndim == 2 else wav
            sf.write(str(p), data, result["sample_rate"])

        _write(voice_path, result["vocals"])
        _write(music_path, result["instrumental"])

        result["paths"] = {"voice": str(voice_path), "music": str(music_path)}
        return result
