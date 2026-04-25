"""Unit tests for :mod:`kube_autotuner.benchmark.client_spec`."""

from __future__ import annotations

import shlex
from typing import Any

import yaml

from kube_autotuner.benchmark.client_spec import build_client_yaml


def _parse(rendered: str) -> dict[str, Any]:
    return yaml.safe_load(rendered)


def _container(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["spec"]["template"]["spec"]["containers"][0]


def _script(doc: dict[str, Any]) -> str:
    """Return the single shell script string passed to ``sh -c``."""
    args = _container(doc)["args"]
    assert len(args) == 1, "expected a single shell-script arg"
    return args[0]


def _iperf_argv(doc: dict[str, Any]) -> list[str]:
    """Tokenise the iperf3 invocation out of the shell wrapper script.

    Returns:
        The argv slice starting at ``iperf3``.
    """
    tokens = shlex.split(_script(doc))
    iperf_idx = tokens.index("iperf3")
    return tokens[iperf_idx:]


def test_build_tcp_default_port():
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="tcp",
        )
    )
    assert doc["kind"] == "Job"
    assert doc["metadata"]["name"] == "iperf3-client-kmain08-p5201"
    node_sel = doc["spec"]["template"]["spec"]["nodeSelector"]
    assert node_sel["kubernetes.io/hostname"] == "kmain08"

    container = _container(doc)
    assert container["image"].startswith("nicolaka/netshoot")
    assert container["command"] == ["sh", "-c"]
    argv = _iperf_argv(doc)
    assert argv[0] == "iperf3"
    assert "-c" in argv
    assert "iperf3-server-kmain07" in argv
    assert "-p" in argv
    assert "5201" in argv
    assert "--json" in argv
    assert "--get-server-output" in argv
    assert "-u" not in argv
    assert "-b" not in argv


def test_build_custom_port():
    doc = _parse(
        build_client_yaml(
            node="kmain09",
            target="kmain07",
            port=5202,
            duration=30,
            omit=5,
            parallel=16,
            mode="tcp",
        )
    )
    assert doc["metadata"]["name"] == "iperf3-client-kmain09-p5202"
    argv = _iperf_argv(doc)
    assert "5202" in argv


def test_build_udp():
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="udp",
        )
    )
    argv = _iperf_argv(doc)
    assert "-u" in argv
    assert argv.index("-b") == argv.index("-u") + 1
    assert argv[argv.index("-b") + 1] == "0"


def test_build_extra_args_appended_after_defaults():
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="tcp",
            extra_args=["--bidir", "-Z"],
        )
    )
    argv = _iperf_argv(doc)
    assert argv.index("-c") < argv.index("--json")
    assert argv[-2:] == ["--bidir", "-Z"]


def test_build_extra_args_preserved_with_udp():
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="udp",
            extra_args=["-Z"],
        )
    )
    argv = _iperf_argv(doc)
    assert "-u" in argv
    assert "-Z" in argv
    assert argv.index("-u") < argv.index("-Z")
    assert argv.index("-b") < argv.index("-Z")


def test_build_udp_bitrate_override():
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="udp",
            extra_args=["-b", "500M"],
        )
    )
    argv = _iperf_argv(doc)
    b_indices = [i for i, a in enumerate(argv) if a == "-b"]
    assert len(b_indices) == 2
    assert argv[b_indices[0] + 1] == "0"
    assert argv[b_indices[1] + 1] == "500M"
    assert b_indices[0] < b_indices[1]


def test_no_barrier_has_no_prologue_and_ends_with_exec_iperf3():
    """Without ``start_at_epoch`` the script is ``set -e`` + ``exec iperf3 ...``."""
    script = _script(
        _parse(
            build_client_yaml(
                node="kmain08",
                target="kmain07",
                port=5201,
                duration=30,
                omit=5,
                parallel=16,
                mode="tcp",
            )
        )
    )
    assert script.startswith("set -e\n")
    assert "NOW=" not in script
    assert "DELTA=" not in script
    assert "sleep" not in script
    assert "exec iperf3" in script
    # Final command line begins with ``exec iperf3 ...`` (the shell
    # replaces itself with iperf3 so the container's PID 1 is iperf3).
    assert script.rstrip().splitlines()[-1].startswith("exec iperf3 ")


def test_barrier_prologue_includes_epoch_literal_and_sleep():
    """With ``start_at_epoch`` the prologue computes and sleeps on DELTA."""
    script = _script(
        _parse(
            build_client_yaml(
                node="kmain08",
                target="kmain07",
                port=5201,
                duration=30,
                omit=5,
                parallel=16,
                mode="tcp",
                start_at_epoch=1_700_000_000,
            )
        )
    )
    assert "NOW=$(date +%s)" in script
    assert "DELTA=$(( 1700000000 - NOW ))" in script
    assert 'if [ "$DELTA" -gt 0 ]; then sleep "$DELTA"; fi' in script
    assert script.rstrip().splitlines()[-1].startswith("exec iperf3 ")


def test_barrier_prologue_preserves_all_flags():
    """The prologue does not drop or reorder any iperf3 flag."""
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="udp",
            extra_args=["-w", "256K", "-Z", "--logfile", "path with spaces"],
            start_at_epoch=1_700_000_000,
        )
    )
    argv = _iperf_argv(doc)
    assert argv[0] == "iperf3"
    assert "iperf3-server-kmain07" in argv
    assert "-u" in argv
    assert "-w" in argv
    assert "256K" in argv
    assert "-Z" in argv
    # shlex round-trip preserves the quoted path with spaces.
    assert argv[-1] == "path with spaces"


def test_no_stderr_redirection_anywhere():
    """The script must never emit to stderr -- kubectl logs merges streams.

    iperf3 JSON is parsed with bare ``json.loads`` on the merged pod log
    stream, so any stderr byte from the prologue would corrupt parsing.
    This regression gate prevents future reintroduction of diagnostic
    ``echo ... 1>&2`` lines.
    """
    script = _script(
        _parse(
            build_client_yaml(
                node="kmain08",
                target="kmain07",
                port=5201,
                duration=30,
                omit=5,
                parallel=16,
                mode="tcp",
                start_at_epoch=1_700_000_000,
            )
        )
    )
    assert "1>&2" not in script
    assert "2>&1" not in script
    assert "echo " not in script
