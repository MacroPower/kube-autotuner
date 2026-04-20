"""Unit tests for :class:`TalosSysctlBackend` (``run_tool``-mocked).

Every ``talosctl`` shell-out routes through
:func:`kube_autotuner.subproc.run_tool`; these tests patch that helper
directly so no real binary is invoked.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import yaml

from kube_autotuner.k8s.client import Kubectl, KubectlError
from kube_autotuner.sysctl.talos import TalosSysctlBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _result(
    rc: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["talosctl"], returncode=rc, stdout=stdout, stderr=stderr
    )


def _make_backend(endpoint: str = "10.5.0.3") -> TalosSysctlBackend:
    return TalosSysctlBackend(
        node="node-1",
        namespace="default",
        kubectl=MagicMock(spec=Kubectl),
        endpoint=endpoint,
    )


def _extract_patch_file_arg(args: Sequence[str]) -> str:
    args_list = list(args)
    idx = args_list.index("-p")
    arg = args_list[idx + 1]
    assert arg.startswith("@"), f"expected @file argument, got {arg!r}"
    return arg[1:]


def _patch_run_tool(
    side_effect: Callable[..., subprocess.CompletedProcess[str]],
):
    return patch("kube_autotuner.sysctl.talos.run_tool", side_effect=side_effect)


def _const_result(
    *, rc: int = 0, stdout: str = "", stderr: str = ""
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def _call(_binary: str, _args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return _result(rc=rc, stdout=stdout, stderr=stderr)

    return _call


class TestApplyCommandShape:
    def test_apply_patches_then_polls(self):
        backend = _make_backend(endpoint="10.5.0.3")
        captured_yaml: dict[str, str] = {}

        readbacks = {
            "/proc/sys/net/core/rmem_max": "16777216\n",
            "/proc/sys/net/core/wmem_max": "1048576\n",
        }

        def fake_run(
            binary: str, args: Sequence[str]
        ) -> subprocess.CompletedProcess[str]:
            assert binary == "talosctl"
            args_list = list(args)
            if "patch" in args_list:
                path = _extract_patch_file_arg(args_list)
                captured_yaml["content"] = Path(path).read_text(encoding="utf-8")
                return _result()
            assert args_list[-2] == "read"
            return _result(stdout=readbacks[args_list[-1]])

        with _patch_run_tool(fake_run) as mock_run:
            backend.apply({
                "net.core.rmem_max": 16777216,
                "net.core.wmem_max": "1048576",
            })

        first_binary, first_args = mock_run.call_args_list[0].args[:2]
        assert first_binary == "talosctl"
        first_args_list = list(first_args)
        assert first_args_list[:6] == [
            "-n",
            "10.5.0.3",
            "patch",
            "mc",
            "--mode=no-reboot",
            "-p",
        ]
        assert first_args_list[6].startswith("@")

        body = yaml.safe_load(captured_yaml["content"])
        assert body == {
            "machine": {
                "sysctls": {
                    "net.core.rmem_max": "16777216",
                    "net.core.wmem_max": "1048576",
                },
            },
        }
        assert all(isinstance(v, str) for v in body["machine"]["sysctls"].values())

        read_calls = mock_run.call_args_list[1:]
        assert read_calls, "expected runtime read-back after patch"
        for call in read_calls:
            call_args = list(call.args[1])
            assert call_args[2] == "read", call_args
        assert not any("reboot" in list(c.args[1]) for c in mock_run.call_args_list)

    def test_apply_issues_one_patch_per_invocation(self):
        backend = _make_backend()

        def fake_run(
            _binary: str, args: Sequence[str]
        ) -> subprocess.CompletedProcess[str]:
            if "patch" in list(args):
                return _result()
            return _result(stdout="16777216\n")

        with _patch_run_tool(fake_run) as mock_run:
            backend.apply({"net.core.rmem_max": "16777216"})
            backend.apply({"net.core.wmem_max": "16777216"})

        patch_calls = [c for c in mock_run.call_args_list if "patch" in list(c.args[1])]
        assert len(patch_calls) == 2
        assert not any("reboot" in list(c.args[1]) for c in mock_run.call_args_list)

    def test_apply_cleans_up_tempfile(self):
        backend = _make_backend()
        captured: dict[str, str] = {}

        def fake_run(
            _binary: str, args: Sequence[str]
        ) -> subprocess.CompletedProcess[str]:
            args_list = list(args)
            if "patch" in args_list:
                captured["path"] = _extract_patch_file_arg(args_list)
                return _result()
            return _result(stdout="16777216\n")

        with _patch_run_tool(fake_run):
            backend.apply({"net.core.rmem_max": "16777216"})

        assert "path" in captured
        assert not Path(captured["path"]).exists()

    def test_apply_cleans_up_tempfile_on_failure(self):
        backend = _make_backend()
        captured: dict[str, str] = {}

        def fake_run(
            _binary: str, args: Sequence[str]
        ) -> subprocess.CompletedProcess[str]:
            args_list = list(args)
            if "patch" in args_list:
                captured["path"] = _extract_patch_file_arg(args_list)
                return _result(rc=1, stderr="permission denied")
            return _result(stdout="16777216\n")

        with _patch_run_tool(fake_run), pytest.raises(RuntimeError):
            backend.apply({"net.core.rmem_max": "16777216"})

        assert "path" in captured
        assert not Path(captured["path"]).exists()

    def test_apply_raises_when_runtime_does_not_propagate(self, monkeypatch):
        monkeypatch.setattr(
            "kube_autotuner.sysctl.talos._APPLY_PROPAGATION_TIMEOUT_SECONDS",
            0.05,
        )
        monkeypatch.setattr(
            "kube_autotuner.sysctl.talos._APPLY_PROPAGATION_POLL_INTERVAL_SECONDS",
            0.01,
        )
        backend = _make_backend()

        def fake_run(
            _binary: str, args: Sequence[str]
        ) -> subprocess.CompletedProcess[str]:
            args_list = list(args)
            if "patch" in args_list or "machineconfig" in args_list:
                return _result(stdout="<diagnostic machineconfig>")
            return _result(stdout="4194304\n")

        with (
            _patch_run_tool(fake_run),
            pytest.raises(RuntimeError, match="did not propagate"),
        ):
            backend.apply({"net.core.rmem_max": "16777216"})


class TestGetCommandShape:
    def test_get_translates_key_to_proc_path(self):
        backend = _make_backend(endpoint="10.5.0.3")
        with _patch_run_tool(_const_result(stdout="16777216\n")) as mock_run:
            result = backend.get(["net.core.rmem_max"])

        binary, args = mock_run.call_args.args[:2]
        assert binary == "talosctl"
        assert list(args) == [
            "-n",
            "10.5.0.3",
            "read",
            "/proc/sys/net/core/rmem_max",
        ]
        assert result == {"net.core.rmem_max": "16777216"}

    def test_get_strips_whitespace(self):
        backend = _make_backend()
        with _patch_run_tool(_const_result(stdout="  212992  \n\n")):
            result = backend.get(["net.core.rmem_max"])
        assert result == {"net.core.rmem_max": "212992"}

    def test_get_issues_only_read_calls(self):
        backend = _make_backend()
        with _patch_run_tool(_const_result(stdout="212992\n")) as mock_run:
            backend.get(["net.core.rmem_max", "net.core.wmem_max"])
        assert mock_run.call_count == 2
        for call in mock_run.call_args_list:
            assert "patch" not in list(call.args[1])


class TestValidation:
    def test_apply_rejects_invalid_key(self):
        backend = _make_backend()
        with pytest.raises(ValueError, match="Invalid sysctl key"):
            backend.apply({"Bad.Key": "1"})

    def test_apply_rejects_invalid_value(self):
        backend = _make_backend()
        with pytest.raises(ValueError, match="Invalid sysctl value"):
            backend.apply({"net.core.rmem_max": "bad;value"})

    def test_get_rejects_invalid_key(self):
        backend = _make_backend()
        with pytest.raises(ValueError, match="Invalid sysctl key"):
            backend.get(["Bad.Key"])


class TestEndpointResolution:
    def test_explicit_endpoint_wins(self):
        kubectl = MagicMock(spec=Kubectl)
        backend = TalosSysctlBackend(
            node="n", kubectl=kubectl, endpoint="explicit.example"
        )
        assert backend.endpoint == "explicit.example"
        kubectl.get_node_internal_ip.assert_not_called()

    def test_kubectl_fallback_when_no_endpoint(self):
        kubectl = MagicMock(spec=Kubectl)
        kubectl.get_node_internal_ip.return_value = "10.0.0.5"
        backend = TalosSysctlBackend(node="n1", kubectl=kubectl)
        assert backend.endpoint == "10.0.0.5"
        kubectl.get_node_internal_ip.assert_called_once_with("n1")

    def test_endpoint_memoised(self):
        kubectl = MagicMock(spec=Kubectl)
        kubectl.get_node_internal_ip.return_value = "10.0.0.5"
        backend = TalosSysctlBackend(node="n1", kubectl=kubectl)
        assert backend.endpoint == "10.0.0.5"
        assert backend.endpoint == "10.0.0.5"
        kubectl.get_node_internal_ip.assert_called_once()

    def test_kubectl_error_wrapped(self):
        kubectl = MagicMock(spec=Kubectl)
        kubectl.get_node_internal_ip.side_effect = KubectlError(
            ["kubectl"], 1, "boom\n"
        )
        backend = TalosSysctlBackend(node="n1", kubectl=kubectl)
        with pytest.raises(RuntimeError, match="boom"):
            _ = backend.endpoint

    def test_empty_internal_ip_raises_with_hint(self):
        kubectl = MagicMock(spec=Kubectl)
        kubectl.get_node_internal_ip.return_value = ""
        backend = TalosSysctlBackend(node="n1", kubectl=kubectl)
        with pytest.raises(RuntimeError, match="endpoint="):
            _ = backend.endpoint


class TestTalosctlErrors:
    def test_missing_talosctl_raises_runtime_error(self):
        backend = _make_backend()
        with (
            patch(
                "kube_autotuner.sysctl.talos.run_tool",
                side_effect=FileNotFoundError("talosctl"),
            ),
            pytest.raises(RuntimeError, match="talosctl not found"),
        ):
            backend.get(["net.core.rmem_max"])

    def test_non_zero_rc_includes_stderr(self):
        backend = _make_backend()
        with (
            _patch_run_tool(_const_result(rc=1, stderr="permission denied")),
            pytest.raises(RuntimeError, match="permission denied"),
        ):
            backend.get(["net.core.rmem_max"])
