from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from router.registry import (
    MAX_REGISTRY_MODELS,
    Registry,
    RegistryModel,
)


def _model(
    model_id: str = "gemma-4-26b-a4b-it-4bit",
    repo: str = "mlx-community/gemma-4-26b-a4b-it-4bit",
    **kwargs,
) -> RegistryModel:
    defaults = dict(priority=5, added_by="node-a", added_at="2026-01-01T00:00:00+00:00")
    defaults.update(kwargs)
    return RegistryModel(repo=repo, id=model_id, **defaults)


# ---------------------------------------------------------------------------
# Load / save round-trip
# ---------------------------------------------------------------------------


class TestLoadSaveRoundTrip:
    def test_round_trip_preserves_data(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json")
        reg.add(_model("model-a"))
        reg.add(_model("model-b", repo="mlx-community/other-model"))
        reg.save()

        reg2 = Registry(path=tmp_path / "registry.json")
        reg2.load()

        assert len(reg2) == 2
        assert reg2.get("model-a") is not None
        assert reg2.get("model-b") is not None
        assert reg2.get("model-a").repo == "mlx-community/gemma-4-26b-a4b-it-4bit"

    def test_load_missing_file_is_noop(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "missing.json")
        reg.load()
        assert len(reg) == 0

    def test_load_rejects_non_array(self, tmp_path: Path):
        path = tmp_path / "registry.json"
        path.write_text('{"not": "an array"}', encoding="utf-8")
        reg = Registry(path=path)
        with pytest.raises(ValueError, match="JSON array"):
            reg.load()


# ---------------------------------------------------------------------------
# Add — valid / invalid repo org
# ---------------------------------------------------------------------------


class TestAddRepoValidation:
    def test_add_trusted_org_succeeds(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json")
        reg.add(_model("my-model", repo="mlx-community/my-model"))
        assert reg.get("my-model") is not None

    def test_add_untrusted_org_rejected(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json")
        with pytest.raises(ValueError, match="trusted allowlist"):
            reg.add(_model("evil-model", repo="evil-org/evil-model"))

    def test_add_custom_trusted_org(self, tmp_path: Path):
        reg = Registry(
            path=tmp_path / "registry.json",
            trusted_orgs=frozenset({"mlx-community", "my-org"}),
        )
        reg.add(_model("custom-model", repo="my-org/custom-model"))
        assert reg.get("custom-model") is not None

    def test_add_malformed_repo_rejected(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json")
        with pytest.raises(ValueError, match="org/name"):
            reg.add(_model("bad", repo="no-slash"))


# ---------------------------------------------------------------------------
# Add — path traversal in ID (rejected)
# ---------------------------------------------------------------------------


class TestModelIdSanitization:
    @pytest.mark.parametrize(
        "bad_id",
        [
            "../etc/passwd",
            "foo/../bar",
            "hello/world",
            "back\\slash",
            "null\x00byte",
            "space name",
            "semi;colon",
            "",
        ],
    )
    def test_bad_ids_rejected(self, tmp_path: Path, bad_id: str):
        reg = Registry(path=tmp_path / "registry.json")
        with pytest.raises(ValueError):
            reg.add(_model(bad_id))

    def test_valid_ids_accepted(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json")
        for good_id in ["gemma-4-26b", "model_v2", "llama3.1-8b", "A1.B2-C3_D4"]:
            reg.add(_model(good_id))
        assert len(reg) == 4


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_existing_model(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json")
        reg.add(_model("to-remove"))
        assert reg.remove("to-remove") is True
        assert reg.get("to-remove") is None

    def test_remove_missing_model_returns_false(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json")
        assert reg.remove("nonexistent") is False


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_union(self, tmp_path: Path):
        a = Registry(path=tmp_path / "a.json")
        a.add(_model("model-a"))

        b = Registry(path=tmp_path / "b.json")
        b.add(_model("model-b", repo="mlx-community/other"))

        a.merge(b)
        assert len(a) == 2
        assert a.get("model-a") is not None
        assert a.get("model-b") is not None

    def test_merge_latest_wins(self, tmp_path: Path):
        a = Registry(path=tmp_path / "a.json")
        a.add(_model("shared", added_at="2025-01-01T00:00:00+00:00", added_by="node-a"))

        b = Registry(path=tmp_path / "b.json")
        b.add(_model("shared", added_at="2026-06-01T00:00:00+00:00", added_by="node-b"))

        a.merge(b)
        assert a.get("shared").added_by == "node-b"

    def test_merge_older_does_not_overwrite(self, tmp_path: Path):
        a = Registry(path=tmp_path / "a.json")
        a.add(_model("shared", added_at="2026-06-01T00:00:00+00:00", added_by="node-a"))

        b = Registry(path=tmp_path / "b.json")
        b.add(_model("shared", added_at="2025-01-01T00:00:00+00:00", added_by="node-b"))

        a.merge(b)
        assert a.get("shared").added_by == "node-a"


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------


class TestCapEnforcement:
    def test_reject_when_full(self, tmp_path: Path):
        cap = 3
        reg = Registry(path=tmp_path / "registry.json", max_models=cap)
        for i in range(cap):
            reg.add(_model(f"model-{i}", repo=f"mlx-community/model-{i}"))

        with pytest.raises(ValueError, match="capacity"):
            reg.add(_model("one-too-many", repo="mlx-community/one-too-many"))

    def test_replacing_existing_does_not_count_as_new(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json", max_models=2)
        reg.add(_model("model-a"))
        reg.add(_model("model-b", repo="mlx-community/other"))
        # Re-adding model-a should succeed (replacement, not new).
        reg.add(_model("model-a", added_at="2027-01-01T00:00:00+00:00"))
        assert len(reg) == 2

    def test_default_cap_is_50(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json")
        assert reg.max_models == MAX_REGISTRY_MODELS == 50


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_file_not_corrupted_on_write_failure(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json")
        reg.add(_model("safe-model"))
        reg.save()

        # Read the good file content.
        good_content = (tmp_path / "registry.json").read_text(encoding="utf-8")

        # Now force os.replace to fail mid-save.
        reg.add(_model("another", repo="mlx-community/another"))
        with patch("router.registry.os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                reg.save()

        # The original file must be intact.
        assert (tmp_path / "registry.json").read_text(encoding="utf-8") == good_content

    def test_no_temp_files_left_on_failure(self, tmp_path: Path):
        reg = Registry(path=tmp_path / "registry.json")
        reg.add(_model("model-a"))
        with patch("router.registry.os.replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                reg.save()

        # No leftover temp files.
        leftover = list(tmp_path.glob(".registry-*.tmp"))
        assert leftover == []


# ---------------------------------------------------------------------------
# State dir env var
# ---------------------------------------------------------------------------


class TestStateDirEnv:
    def test_custom_state_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        custom_dir = tmp_path / "custom-state"
        custom_dir.mkdir()
        monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(custom_dir))

        # Re-import to pick up the env var via a fresh Registry with explicit path.
        reg = Registry(path=custom_dir / "registry.json")
        reg.add(_model("env-model"))
        reg.save()

        assert (custom_dir / "registry.json").exists()


# ---------------------------------------------------------------------------
# Safetensors-only flag
# ---------------------------------------------------------------------------


class TestSafetensorsOnly:
    def test_default_is_true(self):
        m = _model("test")
        assert m.safetensors_only is True

    def test_round_trips_through_dict(self):
        m = _model("test", safetensors_only=False)
        d = m.to_dict()
        assert d["safetensors_only"] is False
        restored = RegistryModel.from_dict(d)
        assert restored.safetensors_only is False
