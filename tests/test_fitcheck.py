from __future__ import annotations

import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from router.fitcheck import FitResult, check_model_fit


def _make_omlx_response(max_model_memory: int, estimated_size: int | None = None) -> dict:
    payload: dict = {"max_model_memory": max_model_memory}
    if estimated_size is not None:
        payload["estimated_size"] = estimated_size
    return payload


def _make_hf_response(safetensors_sizes: list[int] | None = None, extra_files: list[str] | None = None) -> dict:
    siblings = []
    if safetensors_sizes is not None:
        for i, size in enumerate(safetensors_sizes):
            siblings.append({"rfilename": f"model-{i:05d}-of-{len(safetensors_sizes):05d}.safetensors", "size": size})
    for f in extra_files or []:
        siblings.append({"rfilename": f, "size": 100})
    return {"siblings": siblings}


def _mock_urlopen(omlx_resp: dict | None = None, hf_resp: dict | None = None, omlx_error: bool = False, hf_error: bool = False):
    """Return a side_effect function for urllib.request.urlopen."""

    def _side_effect(req, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else str(req)

        if "huggingface.co" in url:
            if hf_error:
                raise urllib.error.URLError("HF unreachable")
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(hf_resp).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        # oMLX
        if omlx_error:
            raise urllib.error.URLError("oMLX unreachable")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(omlx_resp).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    return _side_effect


class TestModelFits:
    def test_model_fits_within_capacity(self):
        omlx = _make_omlx_response(max_model_memory=16_000_000_000)
        hf = _make_hf_response(safetensors_sizes=[4_000_000_000, 4_000_000_000])

        with patch.object(urllib.request, "urlopen", side_effect=_mock_urlopen(omlx_resp=omlx, hf_resp=hf)):
            result = check_model_fit("org/small-model")

        assert result.fits is True
        assert result.model_size_bytes == 8_000_000_000
        assert result.max_memory_bytes == 16_000_000_000
        assert result.has_safetensors is True
        assert "fits" in result.reason.lower()

    def test_model_does_not_fit(self):
        omlx = _make_omlx_response(max_model_memory=8_000_000_000)
        hf = _make_hf_response(safetensors_sizes=[5_000_000_000, 5_000_000_000])

        with patch.object(urllib.request, "urlopen", side_effect=_mock_urlopen(omlx_resp=omlx, hf_resp=hf)):
            result = check_model_fit("org/large-model")

        assert result.fits is False
        assert result.model_size_bytes == 10_000_000_000
        assert result.max_memory_bytes == 8_000_000_000
        assert result.has_safetensors is True
        assert "too large" in result.reason.lower()


class TestOmlxUnreachable:
    def test_returns_fits_false_with_reason(self):
        with patch.object(urllib.request, "urlopen", side_effect=_mock_urlopen(omlx_error=True)):
            result = check_model_fit("org/any-model")

        assert result.fits is False
        assert result.reason == "oMLX not responding"
        assert result.model_size_bytes == 0
        assert result.max_memory_bytes == 0


class TestHuggingFaceUnreachable:
    def test_returns_fits_false_without_fallback(self):
        omlx = _make_omlx_response(max_model_memory=16_000_000_000)

        with patch.object(urllib.request, "urlopen", side_effect=_mock_urlopen(omlx_resp=omlx, hf_error=True)):
            result = check_model_fit("org/any-model")

        assert result.fits is False
        assert result.reason == "Could not estimate model size"
        assert result.max_memory_bytes == 16_000_000_000


class TestNoSafetensors:
    def test_model_with_no_safetensors_files(self):
        omlx = _make_omlx_response(max_model_memory=16_000_000_000)
        hf = _make_hf_response(safetensors_sizes=[], extra_files=["pytorch_model.bin", "config.json"])

        with patch.object(urllib.request, "urlopen", side_effect=_mock_urlopen(omlx_resp=omlx, hf_resp=hf)):
            result = check_model_fit("org/old-model")

        assert result.fits is False
        assert result.has_safetensors is False
        assert result.reason == "No safetensors weights found"


class TestFallbackToOmlxEstimatedSize:
    def test_uses_estimated_size_when_hf_fails(self):
        omlx = _make_omlx_response(max_model_memory=16_000_000_000, estimated_size=6_000_000_000)

        with patch.object(urllib.request, "urlopen", side_effect=_mock_urlopen(omlx_resp=omlx, hf_error=True)):
            result = check_model_fit("org/fallback-model")

        assert result.fits is True
        assert result.model_size_bytes == 6_000_000_000
        assert result.max_memory_bytes == 16_000_000_000
        assert "fits" in result.reason.lower()

    def test_uses_estimated_size_does_not_fit(self):
        omlx = _make_omlx_response(max_model_memory=4_000_000_000, estimated_size=6_000_000_000)

        with patch.object(urllib.request, "urlopen", side_effect=_mock_urlopen(omlx_resp=omlx, hf_error=True)):
            result = check_model_fit("org/fallback-model")

        assert result.fits is False
        assert result.model_size_bytes == 6_000_000_000
        assert "too large" in result.reason.lower()


class TestAuthHeader:
    def test_api_key_sent_as_bearer_token(self):
        omlx = _make_omlx_response(max_model_memory=16_000_000_000)
        hf = _make_hf_response(safetensors_sizes=[1_000_000_000])
        captured_requests: list[urllib.request.Request] = []

        original_side_effect = _mock_urlopen(omlx_resp=omlx, hf_resp=hf)

        def _capturing(req, **kwargs):
            captured_requests.append(req)
            return original_side_effect(req, **kwargs)

        with patch.object(urllib.request, "urlopen", side_effect=_capturing):
            check_model_fit("org/model", omlx_api_key="test-key-123")

        omlx_req = next(r for r in captured_requests if "huggingface" not in r.full_url)
        assert omlx_req.get_header("Authorization") == "Bearer test-key-123"
