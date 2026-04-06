"""Fit check: can this node run a given model?

Estimates whether a model's safetensors weights will fit within the
node's available model memory as reported by the local oMLX instance.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional


@dataclass
class FitResult:
    """Result of a model fit check."""

    fits: bool
    model_size_bytes: int
    max_memory_bytes: int
    reason: str
    has_safetensors: bool


def _get_node_capacity(
    omlx_url: str, omlx_api_key: Optional[str] = None
) -> dict:
    """Fetch node capacity from the local oMLX /v1/models/status endpoint."""
    url = f"{omlx_url.rstrip('/')}/v1/models/status"
    req = urllib.request.Request(url)
    if omlx_api_key:
        req.add_header("Authorization", f"Bearer {omlx_api_key}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _get_hf_model_info(model_repo: str) -> dict:
    """Fetch model metadata from the HuggingFace API."""
    url = f"https://huggingface.co/api/models/{model_repo}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _estimate_safetensors_size(hf_info: dict) -> tuple[int, bool]:
    """Sum safetensors file sizes from HuggingFace model info.

    Returns (total_bytes, has_safetensors).
    """
    siblings = hf_info.get("siblings", [])
    total = 0
    found = False
    for sibling in siblings:
        filename = sibling.get("rfilename", "")
        if filename.endswith(".safetensors"):
            found = True
            size = sibling.get("size", 0)
            if size:
                total += size
    return total, found


def check_model_fit(
    model_repo: str,
    omlx_url: str = "http://127.0.0.1:5741",
    omlx_api_key: Optional[str] = None,
) -> FitResult:
    """Check whether a model will fit on this node.

    Parameters
    ----------
    model_repo:
        HuggingFace model repository identifier (e.g. ``"google/gemma-2b"``).
    omlx_url:
        Base URL for the local oMLX instance.
    omlx_api_key:
        Optional API key for oMLX authentication.

    Returns
    -------
    FitResult
        Dataclass describing whether the model fits and why.
    """
    # Step 1: Get node capacity from oMLX
    try:
        status = _get_node_capacity(omlx_url, omlx_api_key)
    except (urllib.error.URLError, OSError, ValueError):
        return FitResult(
            fits=False,
            model_size_bytes=0,
            max_memory_bytes=0,
            reason="oMLX not responding",
            has_safetensors=True,
        )

    max_memory = status.get("max_model_memory", 0)

    # Step 2: Estimate model size from HuggingFace
    hf_failed = False
    try:
        hf_info = _get_hf_model_info(model_repo)
        model_size, has_safetensors = _estimate_safetensors_size(hf_info)
    except (urllib.error.URLError, OSError, ValueError):
        hf_failed = True
        model_size = 0
        has_safetensors = True  # unknown, assume true

    if hf_failed:
        # Fallback: check if oMLX status has an estimated_size for this model
        estimated_size = status.get("estimated_size")
        if estimated_size:
            model_size = estimated_size
        else:
            return FitResult(
                fits=False,
                model_size_bytes=0,
                max_memory_bytes=max_memory,
                reason="Could not estimate model size",
                has_safetensors=True,
            )

    if not has_safetensors:
        return FitResult(
            fits=False,
            model_size_bytes=0,
            max_memory_bytes=max_memory,
            reason="No safetensors weights found",
            has_safetensors=False,
        )

    # Step 3: Compare
    fits = model_size <= max_memory
    if fits:
        reason = (
            f"Model fits: {model_size / 1e9:.1f} GB model"
            f" within {max_memory / 1e9:.1f} GB capacity"
        )
    else:
        reason = (
            f"Model too large: {model_size / 1e9:.1f} GB model"
            f" exceeds {max_memory / 1e9:.1f} GB capacity"
        )

    return FitResult(
        fits=fits,
        model_size_bytes=model_size,
        max_memory_bytes=max_memory,
        reason=reason,
        has_safetensors=True,
    )
