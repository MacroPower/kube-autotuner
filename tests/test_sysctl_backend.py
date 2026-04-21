"""Tests for ``kube_autotuner.sysctl`` backend, fake, params, and factory.

Covers the ``SysctlBackend`` protocol validators, the in-memory
``FakeSysctlBackend``, ``build_param_space``, and the cluster-free
paths of ``make_sysctl_setter`` / ``make_sysctl_setter_from_env``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from kube_autotuner.experiment import SYSCTL_NAME_RE
from kube_autotuner.models import SysctlParam
from kube_autotuner.sysctl.backend import _validate_sysctl_key  # noqa: PLC2701
from kube_autotuner.sysctl.fake import FakeSysctlBackend
from kube_autotuner.sysctl.params import (
    PARAM_CATEGORIES,
    PARAM_SPACE,
    build_param_space,
)
from kube_autotuner.sysctl.setter import (
    SysctlSetter,
    make_sysctl_setter,
    make_sysctl_setter_from_env,
)
from kube_autotuner.sysctl.talos import TalosSysctlBackend

if TYPE_CHECKING:
    from pathlib import Path

    from kube_autotuner.sysctl.setter import BackendName


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


# Drift guard: the README "Parameter space" section advertises a 27-sysctl
# default broken down into seven categories. If these counts change,
# update the README table in the same commit.
def test_param_space_counts_match_readme():
    assert len(PARAM_SPACE.params) == 27
    expected_counts = {
        "tcp_buffer": 5,
        "congestion": 6,
        "napi": 3,
        "memory": 1,
        "connection": 7,
        "udp": 2,
        "conntrack": 3,
    }
    actual_counts = {cat: len(names) for cat, names in PARAM_CATEGORIES.items()}
    assert actual_counts == expected_counts


def test_param_categories_includes_conntrack_excludes_busy_poll():
    # Dropped in the sysctl parameter-space review: busy_poll / busy_read
    # require application SO_BUSY_POLL opt-in that neither iperf3 nor
    # fortio provide, and the conntrack family is the single biggest gap
    # for a Kubernetes-focused tuner.
    assert "conntrack" in PARAM_CATEGORIES
    assert "busy_poll" not in PARAM_CATEGORIES


def test_conntrack_keys_pass_both_sysctl_regexes():
    # net.netfilter.* must survive both the experiment-level shape check
    # and the backend-level allowlist. Tightening either regex would
    # silently block conntrack tuning.
    key = "net.netfilter.nf_conntrack_max"
    assert SYSCTL_NAME_RE.match(key) is not None
    _validate_sysctl_key(key)  # does not raise


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


class TestMakeSysctlSetter:
    def test_real_returns_real_setter(self):
        assert isinstance(
            make_sysctl_setter(backend="real", node="node1"), SysctlSetter
        )

    def test_talos_returns_talos_backend(self):
        assert isinstance(
            make_sysctl_setter(backend="talos", node="node1"),
            TalosSysctlBackend,
        )

    def test_fake_returns_fake_backend(self, tmp_path: Path):
        backend = make_sysctl_setter(
            backend="fake",
            node="node1",
            fake_state_path=tmp_path / "s.json",
        )
        assert isinstance(backend, FakeSysctlBackend)

    def test_fake_without_state_path_raises(self):
        with pytest.raises(RuntimeError, match="fake_state_path"):
            make_sysctl_setter(backend="fake", node="node1")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown sysctl backend"):
            make_sysctl_setter(backend=cast("BackendName", "garbage"), node="node1")

    def test_explicit_args_not_coupled_to_env(self, monkeypatch):
        # The explicit-arg factory never consults the environment, so
        # setting either env-var name must have no effect.
        monkeypatch.setenv("AUTOTUNER_SYSCTL_BACKEND", "fake")
        monkeypatch.setenv("KUBE_AUTOTUNER_SYSCTL_BACKEND", "fake")
        assert isinstance(
            make_sysctl_setter(backend="real", node="node1"), SysctlSetter
        )


class TestMakeSysctlSetterFromEnv:
    def test_default_real_backend(self):
        assert isinstance(
            make_sysctl_setter_from_env(node="node1", env={}), SysctlSetter
        )

    def test_explicit_real(self):
        backend = make_sysctl_setter_from_env(
            node="node1",
            env={"KUBE_AUTOTUNER_SYSCTL_BACKEND": "real"},
        )
        assert isinstance(backend, SysctlSetter)

    def test_talos(self):
        backend = make_sysctl_setter_from_env(
            node="node1",
            env={"KUBE_AUTOTUNER_SYSCTL_BACKEND": "talos"},
        )
        assert isinstance(backend, TalosSysctlBackend)

    def test_fake_with_state_path(self, tmp_path: Path):
        state = tmp_path / "s.json"
        backend = make_sysctl_setter_from_env(
            node="node1",
            env={
                "KUBE_AUTOTUNER_SYSCTL_BACKEND": "fake",
                "KUBE_AUTOTUNER_SYSCTL_FAKE_STATE": str(state),
            },
        )
        assert isinstance(backend, FakeSysctlBackend)
        assert backend.state_path == state

    def test_fake_without_state_path_raises(self):
        with pytest.raises(RuntimeError, match="KUBE_AUTOTUNER_SYSCTL_FAKE_STATE"):
            make_sysctl_setter_from_env(
                node="node1",
                env={"KUBE_AUTOTUNER_SYSCTL_BACKEND": "fake"},
            )

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown KUBE_AUTOTUNER_SYSCTL_BACKEND"):
            make_sysctl_setter_from_env(
                node="node1",
                env={"KUBE_AUTOTUNER_SYSCTL_BACKEND": "garbage"},
            )

    def test_env_default_falls_back_to_os_environ(self, monkeypatch):
        # When ``env=None`` the helper must read os.environ. Sanity-check
        # that ignores other unrelated env vars.
        monkeypatch.delenv("KUBE_AUTOTUNER_SYSCTL_BACKEND", raising=False)
        monkeypatch.delenv("KUBE_AUTOTUNER_SYSCTL_FAKE_STATE", raising=False)
        assert isinstance(make_sysctl_setter_from_env(node="node1"), SysctlSetter)

    def test_legacy_env_vars_are_ignored(self, monkeypatch, tmp_path: Path):
        # Only the ``KUBE_AUTOTUNER_*`` prefix is honoured; the bare
        # ``AUTOTUNER_*`` prefix must not influence backend selection.
        monkeypatch.delenv("KUBE_AUTOTUNER_SYSCTL_BACKEND", raising=False)
        monkeypatch.setenv("AUTOTUNER_SYSCTL_BACKEND", "fake")
        monkeypatch.setenv("AUTOTUNER_SYSCTL_FAKE_STATE", str(tmp_path / "x.json"))
        assert isinstance(make_sysctl_setter_from_env(node="node1"), SysctlSetter)
