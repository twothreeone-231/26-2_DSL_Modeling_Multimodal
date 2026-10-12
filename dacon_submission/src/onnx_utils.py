"""Shared ONNX Runtime provider-selection helper.

Used by both PolarFormerSeparator and ArtifactNetMusicDetector so the
CUDA-vs-CPU decision (and the "trust what onnxruntime actually loaded, not
what we asked for" check) lives in exactly one place.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def pick_onnx_providers(prefer_gpu: bool) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def resolve_active_provider(session, requested_providers: list[str], component_name: str) -> bool:
    """Returns True if CUDAExecutionProvider is genuinely active for `session`.

    onnxruntime silently falls back to the next provider in the list if a
    requested provider's shared libraries fail to load, instead of raising.
    Trusting the *requested* list (instead of `session.get_providers()`) is a
    real bug we hit once already -- see PolarFormerSeparator's history.
    """
    active_providers = session.get_providers()
    using_gpu = active_providers[0] == "CUDAExecutionProvider"
    if requested_providers[0] == "CUDAExecutionProvider" and not using_gpu:
        logger.warning(
            "%s: requested CUDAExecutionProvider but onnxruntime fell back to %s "
            "(CUDA provider library likely failed to load; often an LD_LIBRARY_PATH "
            "issue -- onnxruntime does a plain dlopen() for libcudnn/libnvrtc/etc and "
            "does not know about pip-installed nvidia-*-cu12 package locations the "
            "way torch's own loader does). Using CPU.",
            component_name,
            active_providers[0],
        )
    return using_gpu
