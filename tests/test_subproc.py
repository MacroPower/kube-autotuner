"""Unit tests for :func:`kube_autotuner.subproc.run_tool`.

Uses ``python -c`` as the external binary so the tests stay portable
and don't depend on ``kubectl``/``talosctl`` being installed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from kube_autotuner.subproc import run_tool


def test_run_tool_happy_path():
    result = run_tool(sys.executable, ["-c", "print('hello')"])
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""


def test_run_tool_captures_stderr():
    result = run_tool(
        sys.executable,
        ["-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
    )
    assert result.returncode == 3
    assert "boom" in result.stderr


def test_run_tool_check_raises_on_nonzero():
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_tool(
            sys.executable,
            ["-c", "import sys; sys.exit(7)"],
            check=True,
        )
    assert excinfo.value.returncode == 7


def test_run_tool_forwards_stdin():
    result = run_tool(
        sys.executable,
        ["-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"],
        input_="hi there",
    )
    assert result.returncode == 0
    assert result.stdout == "HI THERE"


def test_run_tool_missing_binary_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        run_tool("definitely-not-a-real-binary-xyz-42", ["--help"])
