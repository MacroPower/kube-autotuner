"""Real-backend privileged-pod CLI tests.

Read paths run everywhere. The write-path test is marked
``requires_real_sysctl_write`` so it can be opted out on clusters where
privileged pods cannot mutate host sysctls (e.g. Talos Docker); see
``tests/integration/README.md``.
"""

from __future__ import annotations

import contextlib
import json

import pytest
from typer.testing import CliRunner

from kube_autotuner.cli import app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(120),
]


def test_sysctl_get_reads_values(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    node_names: dict[str, str],
    test_namespace: str,
    sysctls_available: None,  # noqa: ARG001 - activation fixture
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "sysctl",
            "get",
            "--node",
            node_names["target"],
            "--namespace",
            test_namespace,
            "-p",
            "net.core.rmem_max",
            "-p",
            "net.core.wmem_max",
        ],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"

    values = json.loads(result.output)
    assert set(values) >= {"net.core.rmem_max", "net.core.wmem_max"}
    for k, v in values.items():
        assert v.isdigit(), f"Expected digit-string for {k}, got {v!r}"


@pytest.mark.requires_real_sysctl_write
def test_sysctl_set_applies_and_sysctl_get_reflects_it(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    node_names: dict[str, str],
    test_namespace: str,
    sysctls_available: None,  # noqa: ARG001 - activation fixture
) -> None:
    runner = CliRunner()
    common = [
        "--node",
        node_names["target"],
        "--namespace",
        test_namespace,
    ]

    get_result = runner.invoke(
        app, ["sysctl", "get", *common, "-p", "net.core.rmem_max"]
    )
    assert get_result.exit_code == 0, f"sysctl get failed:\n{get_result.output}"
    original = json.loads(get_result.output)["net.core.rmem_max"]

    try:
        set_result = runner.invoke(
            app,
            ["sysctl", "set", *common, "-p", "net.core.rmem_max=16777216"],
        )
        assert set_result.exit_code == 0, f"sysctl set failed:\n{set_result.output}"
        assert "Applied 1 sysctl(s)" in set_result.output

        verify_result = runner.invoke(
            app, ["sysctl", "get", *common, "-p", "net.core.rmem_max"]
        )
        assert verify_result.exit_code == 0
        assert json.loads(verify_result.output)["net.core.rmem_max"] == "16777216"
    finally:
        with contextlib.suppress(Exception):
            runner.invoke(
                app,
                ["sysctl", "set", *common, "-p", f"net.core.rmem_max={original}"],
            )


def test_sysctl_set_invalid_param_format_exits_nonzero() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["sysctl", "set", "--node", "irrelevant", "-p", "missing-equals"],
    )
    assert result.exit_code != 0
    assert "Invalid param format" in result.output
