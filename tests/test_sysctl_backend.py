"""Tests for ``kube_autotuner.sysctl.backend``, ``.fake``, and ``.params``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kube_autotuner.models import SysctlParam
from kube_autotuner.sysctl.fake import FakeSysctlBackend
from kube_autotuner.sysctl.params import PARAM_SPACE, build_param_space

if TYPE_CHECKING:
    from pathlib import Path


def test_fake_apply_then_get_returns_set_values(tmp_path: Path):
    backend = FakeSysctlBackend("node1", tmp_path / "s.json")
    backend.apply({"net.core.rmem_max": "16777216"})
    assert backend.get(["net.core.rmem_max"]) == {"net.core.rmem_max": "16777216"}


def test_fake_snapshot_mutate_restore_roundtrip(tmp_path: Path):
    backend = FakeSysctlBackend("node1", tmp_path / "s.json")
    backend.apply({"net.core.rmem_max": "212992"})
    snap = backend.snapshot(["net.core.rmem_max"])
    backend.apply({"net.core.rmem_max": "67108864"})
    backend.restore(snap)
    assert backend.get(["net.core.rmem_max"]) == snap


def test_fake_get_unknown_param_returns_seed_default(tmp_path: Path):
    backend = FakeSysctlBackend("node1", tmp_path / "s.json")
    values = backend.get(["net.core.rmem_max", "kernel.osrelease"])
    assert values["kernel.osrelease"] == "6.1.0-talos"
    assert values["net.core.rmem_max"] == "212992"


def test_fake_defaults_cover_every_param(tmp_path: Path):
    backend = FakeSysctlBackend("node1", tmp_path / "s.json")
    names = PARAM_SPACE.param_names()
    values = backend.get(names)
    for name in names:
        assert values[name], f"missing default for {name}"


def test_fake_invalid_key_raises(tmp_path: Path):
    backend = FakeSysctlBackend("node1", tmp_path / "s.json")
    with pytest.raises(ValueError, match="Invalid sysctl key"):
        backend.apply({"Net.Bad.Key": "1"})


def test_fake_invalid_value_raises(tmp_path: Path):
    backend = FakeSysctlBackend("node1", tmp_path / "s.json")
    with pytest.raises(ValueError, match="Invalid sysctl value"):
        backend.apply({"net.core.rmem_max": "bad;value"})


def test_fake_cross_instance_state_sharing(tmp_path: Path):
    state = tmp_path / "shared.json"
    a = FakeSysctlBackend("node1", state)
    b = FakeSysctlBackend("node1", state)
    a.apply({"net.core.rmem_max": "8388608"})
    assert b.get(["net.core.rmem_max"]) == {"net.core.rmem_max": "8388608"}


def test_fake_lock_is_noop_context_manager(tmp_path: Path):
    backend = FakeSysctlBackend("node1", tmp_path / "s.json")
    with backend.lock() as value:
        assert value is None


def test_build_param_space_canonical_validates():
    ps = build_param_space()
    assert ps.params, "canonical PARAM_SPACE must be non-empty"
    assert ps is not PARAM_SPACE  # builder returns a fresh instance each call


def test_build_param_space_rejects_empty_values():
    bad = [SysctlParam(name="x.y", values=[], param_type="choice")]
    with pytest.raises(ValueError, match="empty values"):
        build_param_space(bad)


def test_build_param_space_rejects_int_param_with_single_value():
    bad = [SysctlParam(name="x.y", values=[42], param_type="int")]
    with pytest.raises(ValueError, match="min < max"):
        build_param_space(bad)


def test_build_param_space_rejects_int_param_with_zero_range():
    bad = [SysctlParam(name="x.y", values=[7, 7], param_type="int")]
    with pytest.raises(ValueError, match="min < max"):
        build_param_space(bad)


def test_build_param_space_accepts_valid_override():
    good = [SysctlParam(name="x.y", values=[1, 2], param_type="int")]
    ps = build_param_space(good)
    assert ps.param_names() == ["x.y"]
