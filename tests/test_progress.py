"""Tests for :mod:`kube_autotuner.progress`."""

from __future__ import annotations

import io
import math
import os
import pty
from typing import TYPE_CHECKING, Any, cast

import pytest
from rich.console import Console
from rich.progress import Progress

from kube_autotuner import progress as progress_module
from kube_autotuner.progress import (
    NullObserver,
    RichProgressObserver,
    make_observer,
)

_IterEtaColumn = progress_module._IterEtaColumn

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


def test_rich_observer_refresh_does_not_force_immediate_paint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: every _refresh must hand Live the renderable with refresh=False.

    Rich's ``Live`` auto-refreshes at 8 Hz; passing ``refresh=True`` on
    every callback stacks event-driven paints on top of the timer's
    paints and produces visible flicker around trial and mode
    boundaries where multiple callbacks fire back-to-back.
    """
    console = _capture_console()
    observer = RichProgressObserver(console)
    calls: list[dict[str, Any]] = []
    with observer:
        assert observer._live is not None
        original_update = observer._live.update

        def spy(renderable: Any, **kwargs: Any) -> None:
            calls.append(dict(kwargs))
            original_update(renderable, **kwargs)

        monkeypatch.setattr(observer._live, "update", spy)
        observer.on_trial_start(0, 2, "sobol", {})
        observer.on_mode_start("tcp", 2)
        observer.on_iteration_start("tcp", 0)
        observer.on_iteration_end("tcp", 0)
        observer.on_trial_complete(0, "sobol", {"throughput": (1e9, 0.0)})
        observer.on_trial_failed(1, RuntimeError("boom"))
    assert calls, "expected at least one Live.update call"
    assert all(not kwargs.get("refresh", False) for kwargs in calls)


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


def test_rich_observer_trial_eta_survives_long_gaps_between_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trials spaced farther apart than Rich's default 30s window still yield an ETA.

    Rich's default ``speed_estimate_period`` of 30s prunes samples older
    than that on every ``update()``. With trials taking minutes, only
    one sample survives the window and ``Task.speed`` is ``None``. The
    observer overrides that by setting ``speed_estimate_period`` on the
    trials ``Progress``.
    """
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 1000.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._trials, "get_time", _fake)
    monkeypatch.setattr(observer._iters, "get_time", _fake)
    with observer:
        observer.on_trial_start(0, 5, "sobol", {})
        clock["t"] += 120.0
        observer.on_trial_complete(0, "sobol", {"throughput": (1e9, 0.0)})
        clock["t"] += 120.0  # >> 30s default speed window
        observer.on_trial_complete(1, "sobol", {"throughput": (1e9, 0.0)})
        trial_task = observer._trials.tasks[0]
        assert trial_task.speed is not None
        assert trial_task.time_remaining is not None
        assert trial_task.time_remaining > 0


def _find_iter_eta_column(observer: RichProgressObserver) -> _IterEtaColumn:
    for column in observer._iters.columns:
        if isinstance(column, _IterEtaColumn):
            return column
    msg = "observer._iters is missing an _IterEtaColumn"
    raise AssertionError(msg)


def test_rich_observer_iteration_eta_survives_mode_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Iteration-bar ETA persists across ``on_mode_start`` resets."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 1000.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._trials, "get_time", _fake)
    monkeypatch.setattr(observer._iters, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)
    with observer:
        observer.on_mode_start("tcp", 3)
        observer.on_iteration_start("tcp", 0)
        clock["t"] += 60.0
        observer.on_iteration_end("tcp", 0)
        observer.on_iteration_start("tcp", 1)
        clock["t"] += 60.0
        observer.on_iteration_end("tcp", 1)
        assert observer._iter_count == 2
        assert observer._iter_total_seconds == pytest.approx(120.0)
        column = _find_iter_eta_column(observer)
        # Mid-mode: 1 iteration remaining * 60s mean = 0:01:00.
        assert column.render(observer._iters.tasks[0]).plain == "0:01:00"
        clock["t"] += 60.0
        observer.on_mode_start("udp", 3)
        # History survives the reset.
        assert observer._iter_count == 2
        # 3 remaining * 60s mean = 0:03:00 — no "-:--:--" flash.
        assert column.render(observer._iters.tasks[0]).plain == "0:03:00"


def test_iter_eta_column_renders_dashes_without_history() -> None:
    """The column falls back to ``-:--:--`` when the history is empty."""
    column = _IterEtaColumn(lambda: (0, 0.0))
    progress = Progress(column)
    progress.add_task("x", total=3, completed=0)
    rendered = column.render(progress.tasks[0])
    assert rendered.plain == "-:--:--"


def test_rich_observer_iteration_history_survives_across_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History carries across a full trial boundary (tcp → udp → tcp)."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 500.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._trials, "get_time", _fake)
    monkeypatch.setattr(observer._iters, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)

    def _do_iter(mode: str, iteration: int, duration: float) -> None:
        observer.on_iteration_start(mode, iteration)
        clock["t"] += duration
        observer.on_iteration_end(mode, iteration)

    with observer:
        observer.on_mode_start("tcp", 2)
        _do_iter("tcp", 0, 30.0)
        _do_iter("tcp", 1, 30.0)
        observer.on_mode_start("udp", 2)
        _do_iter("udp", 0, 30.0)
        _do_iter("udp", 1, 30.0)
        observer.on_mode_start("tcp", 3)
        assert observer._iter_count == 4
        assert observer._iter_total_seconds == pytest.approx(120.0)
        column = _find_iter_eta_column(observer)
        # Fresh trial-2 tcp task: 3 remaining * 30s mean = 0:01:30.
        assert column.render(observer._iters.tasks[0]).plain == "0:01:30"


def test_rich_observer_aborted_iteration_does_not_pollute_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A started-but-never-ended iteration is dropped at the next mode start."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._trials, "get_time", _fake)
    monkeypatch.setattr(observer._iters, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)
    with observer:
        observer.on_mode_start("tcp", 3)
        observer.on_iteration_start("tcp", 0)
        clock["t"] += 9999.0  # aborted iteration, huge elapsed
        observer.on_mode_start("udp", 3)  # no matching on_iteration_end
        observer.on_iteration_start("udp", 0)
        clock["t"] += 45.0
        observer.on_iteration_end("udp", 0)
    assert observer._iter_count == 1
    assert observer._iter_total_seconds == pytest.approx(45.0)


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
