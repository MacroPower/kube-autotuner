"""Tests for :mod:`kube_autotuner.progress`."""

from __future__ import annotations

import io
import math
import os
import pty
from typing import TYPE_CHECKING, Any, cast

import pytest
from rich.console import Console

from kube_autotuner import progress as progress_module
from kube_autotuner.progress import (
    NullObserver,
    RichProgressObserver,
    make_observer,
)

_TtyEchoSuppressor = progress_module._TtyEchoSuppressor

if TYPE_CHECKING:
    from kube_autotuner.progress import ProgressObserver


def _capture_console() -> Console:
    # Force-render to an in-memory buffer so assertions can inspect output.
    return Console(
        file=io.StringIO(),
        force_terminal=True,
        width=120,
        color_system=None,
    )


def test_null_observer_is_protocol_compliant() -> None:
    obs: ProgressObserver = NullObserver()
    with obs as active:
        assert active is obs
        active.on_trial_start(0, 5, "sobol", {"net.core.rmem_max": "1048576"})
        active.on_mode_start("tcp", 3)
        active.on_iteration_start("tcp", 0)
        active.on_iteration_end("tcp", 0)
        active.on_trial_complete(
            0,
            "sobol",
            {"throughput": (9_400_000_000.0, 1e6), "cpu": (42.0, 0.5)},
        )
        active.on_trial_failed(1, RuntimeError("boom"))


def test_rich_observer_renders_bars_and_table() -> None:
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_trial_start(0, 3, "sobol", {"net.core.rmem_max": "1048576"})
        observer.on_mode_start("tcp", 2)
        observer.on_iteration_start("tcp", 0)
        observer.on_iteration_end("tcp", 0)
        observer.on_trial_complete(
            0,
            "sobol",
            {
                "throughput": (9_412_000_000.0, 1e7),
                "cpu": (42.1, 0.4),
                "retransmit_rate": (1.2e-7, 1e-9),
            },
        )
        observer.on_trial_complete(
            1,
            "bayesian",
            {
                "throughput": (9_188_000_000.0, 1e7),
                "cpu": (38.4, 0.3),
                "retransmit_rate": (float("nan"), float("nan")),
            },
        )
        observer.on_trial_failed(2, RuntimeError("boom"))
    output = cast("io.StringIO", console.file).getvalue()
    assert "Trials" in output
    assert "Best so far" in output
    assert "9,412.0 Mbps" in output
    assert "0.12" in output  # retx/MB rendered


def test_rich_observer_handles_missing_metrics() -> None:
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_trial_start(0, 1, "sobol", {})
        observer.on_trial_complete(0, "sobol", {})
    # No KeyError; row folded in with zeros / NaN.
    assert observer._top
    assert math.isnan(observer._top[0].retx_per_mb)


def test_rich_observer_tracks_top_n() -> None:
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        for i, tp in enumerate([1, 9, 5, 7, 3, 8, 2], start=0):
            observer.on_trial_start(i, 10, "bayesian", {})
            observer.on_trial_complete(
                i,
                "bayesian",
                {
                    "throughput": (tp * 1e9, 0.0),
                    "cpu": (10.0, 0.0),
                    "retransmit_rate": (0.0, 0.0),
                },
            )
    # Keep the top five by throughput, descending.
    top_mbps = [int(r.throughput_mbps / 1000) for r in observer._top]
    assert top_mbps == [9, 8, 7, 5, 3]


def test_make_observer_enabled_returns_rich_implementation() -> None:
    console = _capture_console()
    obs = make_observer(enabled=True, console=console)
    assert isinstance(obs, RichProgressObserver)


def test_make_observer_disabled_returns_null() -> None:
    console = _capture_console()
    obs = make_observer(enabled=False, console=console)
    assert isinstance(obs, NullObserver)


# --- _TtyEchoSuppressor coverage -------------------------------------------


class _RecordingTty:
    """Stand-in for the ``tty`` module that records setcbreak calls."""

    def __init__(self, sentinel: list[Any]) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._sentinel = sentinel

    def setcbreak(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.calls.append((args, kwargs))
        return self._sentinel


class _RecordingTermios:
    """Stand-in for the ``termios`` module that records tcsetattr calls."""

    TCSAFLUSH = 2
    ECHO = 0o10
    ICANON = 0o2
    ISIG = 0o1

    class error(Exception):  # noqa: N801, N818 mirrors termios.error
        """Mimic ``termios.error``."""

    def __init__(self) -> None:
        self.set_calls: list[tuple[Any, ...]] = []

    def tcsetattr(self, *args: Any, **kwargs: Any) -> None:
        self.set_calls.append((args, kwargs))


def test_tty_echo_suppressor_noop_when_stdin_not_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tty = _RecordingTty(sentinel=[])
    fake_termios = _RecordingTermios()
    monkeypatch.setattr(progress_module, "tty", fake_tty)
    monkeypatch.setattr(progress_module, "termios", fake_termios)
    monkeypatch.setattr(progress_module.os, "isatty", lambda _fd: False)

    class _FakeStdin:
        def fileno(self) -> int:
            return 99

    monkeypatch.setattr(progress_module.sys, "stdin", _FakeStdin())

    with _TtyEchoSuppressor():
        pass

    assert fake_tty.calls == []
    assert fake_termios.set_calls == []


def test_tty_echo_suppressor_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel: list[Any] = ["SAVED"]
    fake_tty = _RecordingTty(sentinel=sentinel)
    fake_termios = _RecordingTermios()
    monkeypatch.setattr(progress_module, "tty", fake_tty)
    monkeypatch.setattr(progress_module, "termios", fake_termios)
    monkeypatch.setattr(progress_module.os, "isatty", lambda _fd: True)

    class _FakeStdin:
        def fileno(self) -> int:
            return 7

    monkeypatch.setattr(progress_module.sys, "stdin", _FakeStdin())

    with _TtyEchoSuppressor():
        assert fake_tty.calls == [((7, fake_termios.TCSAFLUSH), {})]
        assert fake_termios.set_calls == []

    assert fake_termios.set_calls == [((7, fake_termios.TCSAFLUSH, sentinel), {})]


def test_tty_echo_suppressor_preserves_isig_on_real_pty() -> None:
    pytest.importorskip("termios")
    import sys  # noqa: PLC0415 per-test stdin swap
    import termios  # noqa: PLC0415 required to read back the pty state

    master, slave = pty.openpty()
    try:
        saved = termios.tcgetattr(slave)
        saved_lflag = saved[3]

        class _SlaveStdin:
            def fileno(self) -> int:
                return slave

        original_stdin = sys.stdin
        sys.stdin = cast("Any", _SlaveStdin())
        try:
            with _TtyEchoSuppressor():
                inside = termios.tcgetattr(slave)
                inside_lflag = inside[3]
                assert inside_lflag & termios.ECHO == 0
                assert inside_lflag & termios.ICANON == 0
                assert inside_lflag & termios.ISIG == saved_lflag & termios.ISIG
            after = termios.tcgetattr(slave)
            assert after[3] == saved_lflag
        finally:
            sys.stdin = original_stdin
    finally:
        os.close(master)
        os.close(slave)


def test_rich_observer_restores_echo_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel: list[Any] = ["SAVED"]
    fake_tty = _RecordingTty(sentinel=sentinel)
    fake_termios = _RecordingTermios()
    monkeypatch.setattr(progress_module, "tty", fake_tty)
    monkeypatch.setattr(progress_module, "termios", fake_termios)
    monkeypatch.setattr(progress_module.os, "isatty", lambda _fd: True)

    class _FakeStdin:
        def fileno(self) -> int:
            return 11

    monkeypatch.setattr(progress_module.sys, "stdin", _FakeStdin())

    console = _capture_console()
    observer = RichProgressObserver(console)
    with pytest.raises(RuntimeError, match="boom"), observer:
        raise RuntimeError("boom")

    assert fake_termios.set_calls == [((11, fake_termios.TCSAFLUSH, sentinel), {})]


def test_rich_observer_unwinds_echo_when_live_enter_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel: list[Any] = ["SAVED"]
    fake_tty = _RecordingTty(sentinel=sentinel)
    fake_termios = _RecordingTermios()
    monkeypatch.setattr(progress_module, "tty", fake_tty)
    monkeypatch.setattr(progress_module, "termios", fake_termios)
    monkeypatch.setattr(progress_module.os, "isatty", lambda _fd: True)

    class _FakeStdin:
        def fileno(self) -> int:
            return 13

    monkeypatch.setattr(progress_module.sys, "stdin", _FakeStdin())

    def _boom(_self: Any) -> None:
        msg = "live boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(progress_module.Live, "__enter__", _boom)

    console = _capture_console()
    observer = RichProgressObserver(console)
    with pytest.raises(RuntimeError, match="live boom"), observer:
        pass

    assert observer._live is None
    assert observer._echo is None
    assert fake_termios.set_calls == [((13, fake_termios.TCSAFLUSH, sentinel), {})]


def test_tty_echo_suppressor_noop_when_termios_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_module, "termios", None)
    monkeypatch.setattr(progress_module, "tty", None)

    sentinel_called = False

    class _FakeStdin:
        def fileno(self) -> int:
            nonlocal sentinel_called
            sentinel_called = True
            return 0

    monkeypatch.setattr(progress_module.sys, "stdin", _FakeStdin())

    with _TtyEchoSuppressor():
        pass

    assert sentinel_called is False
