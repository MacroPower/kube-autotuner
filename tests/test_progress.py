"""Tests for :mod:`kube_autotuner.progress`."""

from __future__ import annotations

import io
import math
import operator
import os
import pty
from typing import TYPE_CHECKING, Any, cast

import pytest
from rich.console import Console
from rich.progress import Progress

from kube_autotuner import progress as progress_module
from kube_autotuner.experiment import ObjectivesSection
from kube_autotuner.progress import (
    NullObserver,
    RichProgressObserver,
    make_observer,
)

_HistoryEtaColumn = progress_module._HistoryEtaColumn
_STAGES_PER_ITERATION = progress_module._STAGES_PER_ITERATION

_TtyEchoSuppressor = progress_module._TtyEchoSuppressor

if TYPE_CHECKING:
    from kube_autotuner.models import TrialResult
    from kube_autotuner.progress import ProgressObserver


def _stub_trial(
    phase: str,
    *,
    trial_id: str | None = None,
    parent_trial_id: str | None = None,
) -> TrialResult:
    """Build a minimal :class:`TrialResult` for observer tests.

    Returns:
        A trial record stamped with ``phase``, ``parent_trial_id``,
        and (optionally) a fixed ``trial_id``.
    """
    from kube_autotuner.models import (  # noqa: PLC0415
        BenchmarkConfig,
        NodePair,
        TrialResult,
    )

    tr = TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        phase=phase,  # ty: ignore[invalid-argument-type]
        parent_trial_id=parent_trial_id,
    )
    if trial_id is not None:
        tr.trial_id = trial_id
    return tr


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
        active.on_benchmark_start(3)
        active.on_iteration_start(0)
        active.on_stage_start("bw-tcp", 0)
        active.on_stage_end("bw-tcp", 0)
        active.on_iteration_end(0)
        active.on_trial_complete(
            0,
            _stub_trial("sobol"),
            {"tcp_throughput": (9_400_000_000.0, 1e6), "udp_jitter": (0.05, 0.01)},
        )
        active.on_trial_failed(1, RuntimeError("boom"))


def test_rich_observer_renders_bars_and_table() -> None:
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_trial_start(0, 3, "sobol", {"net.core.rmem_max": "1048576"})
        observer.on_benchmark_start(2)
        observer.on_iteration_start(0)
        observer.on_iteration_end(0)
        observer.on_trial_complete(
            0,
            _stub_trial("sobol"),
            {
                "tcp_throughput": (9_412_000_000.0, 1e7),
                "tcp_retransmit_rate": (0.12, 1e-3),
                "udp_jitter": (0.000128, 1e-5),
            },
        )
        observer.on_trial_complete(
            1,
            _stub_trial("bayesian"),
            {
                "tcp_throughput": (9_188_000_000.0, 1e7),
                "tcp_retransmit_rate": (float("nan"), float("nan")),
                "udp_jitter": (float("nan"), float("nan")),
            },
        )
        observer.on_trial_failed(2, RuntimeError("boom"))
    output = cast("io.StringIO", console.file).getvalue()
    assert "Trials" in output
    assert "Best so far" in output
    assert "9,412.0 Mbps" in output
    assert "0.12" in output  # retx/GB rendered
    assert "jitter us" in output  # jitter column header picks us (128 us)
    assert "128" in output  # jitter value rendered as us coefficient


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
        observer.on_benchmark_start(2)
        observer.on_iteration_start(0)
        observer.on_iteration_end(0)
        observer.on_trial_complete(
            0,
            _stub_trial("sobol"),
            {"tcp_throughput": (1e9, 0.0)},
        )
        observer.on_trial_failed(1, RuntimeError("boom"))
    assert calls, "expected at least one Live.update call"
    assert all(not kwargs.get("refresh", False) for kwargs in calls)


def test_rich_observer_handles_missing_metrics() -> None:
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_trial_start(0, 1, "sobol", {})
        observer.on_trial_complete(0, _stub_trial("sobol"), {})
    # No KeyError; row folded in with zeros / NaN.
    assert observer._top
    assert math.isnan(observer._top[0].retx_per_gb)


def _prior_trial(bps: float, *, phase: str | None = None) -> Any:
    from datetime import UTC, datetime  # noqa: PLC0415

    from kube_autotuner.models import (  # noqa: PLC0415
        BenchmarkConfig,
        BenchmarkResult,
        NodePair,
        TrialResult,
    )

    return TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": int(bps)},
        config=BenchmarkConfig(),
        phase=phase,  # ty: ignore[invalid-argument-type]
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=bps,
                retransmits=5,
                bytes_sent=1_000_000_000,
                iteration=0,
            ),
        ],
    )


def test_null_seed_history_is_noop() -> None:
    obs = NullObserver()
    obs.seed_history([_prior_trial(1e9)], n_sobol=2)
    # No attrs to assert on; the fact that no exception was raised is the test.


def test_rich_seed_history_populates_top_and_phase() -> None:
    pytest.importorskip("ax")
    console = _capture_console()
    observer = RichProgressObserver(console)
    priors = [
        _prior_trial(5e9, phase="sobol"),
        _prior_trial(9e9, phase="sobol"),
        _prior_trial(7e9, phase="bayesian"),
    ]
    observer.seed_history(priors, n_sobol=2)
    # Top is sorted by throughput descending and capped.
    tps = [r.throughput_mbps for r in observer._top]
    assert tps == sorted(tps, reverse=True)
    assert len(observer._top) == 3
    # Phase label reflects number of priors vs n_sobol (3 priors, n_sobol=2
    # → "bayesian").
    assert observer._phase == "bayesian"
    # Trial-timing counters are untouched by seed_history.
    assert observer._trial_count == 0
    assert observer._trial_total_seconds == pytest.approx(0.0)


def test_rich_seed_history_infers_sobol_phase_when_under_budget() -> None:
    pytest.importorskip("ax")
    console = _capture_console()
    observer = RichProgressObserver(console)
    observer.seed_history([_prior_trial(1e9, phase="sobol")], n_sobol=5)
    assert observer._phase == "sobol"


def test_seed_history_importable_without_ax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy import inside seed_history fires only at call time.

    ``RichProgressObserver`` itself must be constructible and ``_top``
    assertable without ``ax-platform`` being importable at module
    load time. This pins the rule that ``progress`` stays optimize-
    free for ``task completions``.
    """
    import sys  # noqa: PLC0415

    # Block every `ax` import path so module-level code that tried to
    # pull it in would fail loudly.
    blocked = [
        name for name in list(sys.modules) if name == "ax" or name.startswith("ax.")
    ]
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)  # block re-imports

    console = _capture_console()
    observer = RichProgressObserver(console)
    # Constructing + using the non-seed methods must still work.
    observer.on_trial_complete(
        0,
        _stub_trial("sobol"),
        {"tcp_throughput": (1e9, 0.0)},
    )
    assert observer._top


def test_rich_observer_tracks_top_n() -> None:
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        for i, tp in enumerate([1, 9, 5, 7, 3, 8, 2], start=0):
            observer.on_trial_start(i, 10, "bayesian", {})
            observer.on_trial_complete(
                i,
                _stub_trial("bayesian"),
                {
                    "tcp_throughput": (tp * 1e9, 0.0),
                    "tcp_retransmit_rate": (0.0, 0.0),
                },
            )
    # Keep the top five by throughput, descending.
    top_mbps = [int(r.throughput_mbps / 1000) for r in observer._top]
    assert top_mbps == [9, 8, 7, 5, 3]


def test_rich_observer_trial_eta_survives_long_gaps_between_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trials spaced farther apart than Rich's default 30s window still yield an ETA.

    History-backed ETA is driven by observer-owned timestamps rather
    than Rich's task-local sample deque, so gaps longer than Rich's
    default 30s ``speed_estimate_period`` cannot prune the signal.
    """
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 1000.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)
    with observer:
        observer.on_trial_start(0, 5, "sobol", {})
        clock["t"] += 120.0
        observer.on_trial_complete(
            0,
            _stub_trial("sobol"),
            {"tcp_throughput": (1e9, 0.0)},
        )
        clock["t"] += 120.0  # >> 30s default speed window
        observer.on_trial_start(1, 5, "sobol", {})
        clock["t"] += 120.0
        observer.on_trial_complete(
            1,
            _stub_trial("sobol"),
            {"tcp_throughput": (1e9, 0.0)},
        )
        column = _find_eta_column(observer)
        rendered = column.render(_task_for(observer, observer._trial_task_id)).plain
        assert rendered != "-:--:--"


def _find_eta_column(observer: RichProgressObserver) -> _HistoryEtaColumn:
    for column in observer._progress.columns:
        if isinstance(column, _HistoryEtaColumn):
            return column
    msg = "observer._progress is missing a _HistoryEtaColumn"
    raise AssertionError(msg)


def _task_for(observer: RichProgressObserver, task_id: object) -> Any:
    """Return the Rich task in ``observer._progress`` with ``task_id``."""
    return next(t for t in observer._progress.tasks if t.id == task_id)


def test_rich_observer_iteration_eta_survives_stage_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Iteration-bar ETA persists across stage boundaries within an iteration."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 1000.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)
    with observer:
        observer.on_benchmark_start(6)
        observer.on_iteration_start(0)
        clock["t"] += 60.0
        observer.on_iteration_end(0)
        observer.on_iteration_start(1)
        clock["t"] += 60.0
        observer.on_iteration_end(1)
        assert observer._iter_count == 2
        assert observer._iter_total_seconds == pytest.approx(120.0)
        column = _find_eta_column(observer)
        iter_task = _task_for(observer, observer._iter_task_id)
        # Mid-trial: 4 iterations remaining * 60s mean = 0:04:00.
        assert column.render(iter_task).plain == "0:04:00"
        clock["t"] += 60.0
        # Advance into a stage transition without completing an iteration.
        observer.on_stage_start("bw-udp", 2)
        assert observer._iter_count == 2
        # History preserved; still 4 remaining * 60s mean = 0:04:00.
        assert column.render(iter_task).plain == "0:04:00"


def test_iter_eta_column_renders_dashes_without_history() -> None:
    """The column falls back to ``-:--:--`` when the history is empty."""
    column = _HistoryEtaColumn(lambda _task_id: (0, 0.0, 0.0))
    progress = Progress(column)
    progress.add_task("x", total=3, completed=0)
    rendered = column.render(progress.tasks[0])
    assert rendered.plain == "-:--:--"


def test_rich_observer_trial_eta_ticks_down_mid_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-flight trial elapsed reduces the ETA every refresh, not just on completion."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)
    with observer:
        observer.on_trial_start(0, 3, "sobol", {})
        clock["t"] += 120.0
        observer.on_trial_complete(
            0,
            _stub_trial("sobol"),
            {"tcp_throughput": (1e9, 0.0)},
        )
        observer.on_trial_start(1, 3, "sobol", {})
        column = _find_eta_column(observer)
        task = _task_for(observer, observer._trial_task_id)
        # Trial 1 just started: 2 remaining * 120s mean = 0:04:00.
        assert column.render(task).plain == "0:04:00"
        clock["t"] += 30.0
        # 30s in: 1*120 + max(0, 120-30) = 210s = 0:03:30.
        assert column.render(task).plain == "0:03:30"
        clock["t"] += 60.0
        # 90s in: 1*120 + max(0, 120-90) = 150s = 0:02:30.
        assert column.render(task).plain == "0:02:30"
        clock["t"] += 60.0
        # 150s in (over mean): floors at 1*120 = 0:02:00.
        assert column.render(task).plain == "0:02:00"


def test_rich_observer_trial_eta_available_after_first_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Trials-bar ETA is concrete after a single completion.

    Rich's ``TimeRemainingColumn`` requires two samples to compute
    ``Task.speed``. The history-backed column reads observer-owned
    timestamps, so one completed trial is enough.
    """
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)
    with observer:
        observer.on_trial_start(0, 5, "sobol", {})
        clock["t"] += 120.0
        observer.on_trial_complete(
            0,
            _stub_trial("sobol"),
            {"tcp_throughput": (1e9, 0.0)},
        )
        column = _find_eta_column(observer)
        trial_task = _task_for(observer, observer._trial_task_id)
        # 4 remaining trials * 120s mean = 0:08:00.
        assert column.render(trial_task).plain == "0:08:00"


def test_rich_observer_trial_eta_counts_failed_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed trials contribute wall time to the trial ETA mean."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)
    with observer:
        observer.on_trial_start(0, 5, "sobol", {})
        clock["t"] += 60.0
        observer.on_trial_failed(0, RuntimeError("x"))
        observer.on_trial_start(1, 5, "sobol", {})
        clock["t"] += 120.0
        observer.on_trial_complete(
            1,
            _stub_trial("sobol"),
            {"tcp_throughput": (1e9, 0.0)},
        )
        assert observer._trial_count == 2
        assert observer._trial_total_seconds == pytest.approx(180.0)
        column = _find_eta_column(observer)
        trial_task = _task_for(observer, observer._trial_task_id)
        # 3 remaining * 90s mean = 0:04:30.
        assert column.render(trial_task).plain == "0:04:30"


def test_rich_observer_iteration_history_survives_across_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History carries across a full trial boundary."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 500.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)

    def _do_iter(iteration: int, duration: float) -> None:
        observer.on_iteration_start(iteration)
        clock["t"] += duration
        observer.on_iteration_end(iteration)

    with observer:
        # Trial 1: 4 iterations at 30s each.
        observer.on_benchmark_start(4)
        for i in range(4):
            _do_iter(i, 30.0)
        # Trial 2: 3 iterations remaining.
        observer.on_benchmark_start(3)
        assert observer._iter_count == 4
        assert observer._iter_total_seconds == pytest.approx(120.0)
        column = _find_eta_column(observer)
        iter_task = _task_for(observer, observer._iter_task_id)
        # Fresh trial-2 bar: 3 remaining * 30s mean = 0:01:30.
        assert column.render(iter_task).plain == "0:01:30"


def test_rich_observer_aborted_iteration_does_not_pollute_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A started-but-never-ended iteration is dropped at the next benchmark start."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)
    with observer:
        observer.on_benchmark_start(3)
        observer.on_iteration_start(0)
        clock["t"] += 9999.0  # aborted iteration, huge elapsed
        observer.on_benchmark_start(3)  # fresh trial; no matching on_iteration_end
        observer.on_iteration_start(0)
        clock["t"] += 45.0
        observer.on_iteration_end(0)
    assert observer._iter_count == 1
    assert observer._iter_total_seconds == pytest.approx(45.0)


def test_rich_observer_iteration_bar_spans_whole_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The iteration bar is sized to ``iterations`` and spans stage transitions."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)

    def _do_iter(iteration: int) -> None:
        observer.on_iteration_start(iteration)
        observer.on_stage_start("bw-tcp", iteration)
        clock["t"] += 10.0
        observer.on_stage_end("bw-tcp", iteration)
        observer.on_iteration_end(iteration)

    with observer:
        observer.on_benchmark_start(6)
        for i in range(4):
            _do_iter(i)
        # Mid-trial: open the bw-udp stage of iteration 4 without
        # closing it. The stage label lives on the stage bar now;
        # the iteration bar keeps the plain "Current" description.
        observer.on_iteration_start(4)
        observer.on_stage_start("bw-udp", 4)
        task = _task_for(observer, observer._iter_task_id)
        assert task.total == 6
        assert task.completed == 4
        assert task.description == "Current"
        stage_task = _task_for(observer, observer._stage_task_id)
        assert stage_task.description == "Stage [bw-udp]"


def test_rich_observer_iteration_bar_resets_between_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``on_benchmark_start`` re-scopes the bar while history is preserved."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)

    def _do_iter(iteration: int) -> None:
        observer.on_iteration_start(iteration)
        clock["t"] += 15.0
        observer.on_iteration_end(iteration)

    with observer:
        observer.on_benchmark_start(4)
        for i in range(4):
            _do_iter(i)
        # Trial 2 begins.
        observer.on_benchmark_start(4)
        task = _task_for(observer, observer._iter_task_id)
        assert task.total == 4
        assert task.completed == 0
        # History survives across trials.
        assert observer._iter_count == 4


def test_rich_observer_iteration_eta_covers_remaining_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ETA reflects iterations that have not yet started."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)

    def _do_iter(iteration: int) -> None:
        observer.on_iteration_start(iteration)
        clock["t"] += 60.0
        observer.on_iteration_end(iteration)

    with observer:
        observer.on_benchmark_start(6)
        for i in range(3):
            _do_iter(i)
        column = _find_eta_column(observer)
        iter_task = _task_for(observer, observer._iter_task_id)
        # 3 iterations remain at 60s each.
        assert column.render(iter_task).plain == "0:03:00"


def test_rich_observer_iteration_bar_resets_after_mid_trial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-trial failure leaves the bar partial; the next trial re-scopes it."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)

    def _do_iter(iteration: int) -> None:
        observer.on_iteration_start(iteration)
        clock["t"] += 20.0
        observer.on_iteration_end(iteration)

    with observer:
        # Trial 1 begins but fails after 2 of 6 iterations.
        observer.on_benchmark_start(6)
        _do_iter(0)
        _do_iter(1)
        # Simulate runner raising out: no further callbacks for this trial.

        # Trial 2 starts fresh.
        observer.on_benchmark_start(6)
        task = _task_for(observer, observer._iter_task_id)
        assert task.total == 6
        assert task.completed == 0
        # History survives the failed trial.
        assert observer._iter_count == 2


def test_rich_observer_stage_bar_created_on_benchmark_start() -> None:
    """``on_benchmark_start`` adds a ``Stage`` task sized to 4 sub-stages."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_benchmark_start(2)
        assert observer._stage_task_id is not None
        task = _task_for(observer, observer._stage_task_id)
        assert task.total == _STAGES_PER_ITERATION
        assert task.completed == 0
        assert task.description == "Stage"


def test_rich_observer_stage_bar_honors_custom_stages() -> None:
    """Custom ``stages`` sizes and clamps the Stage bar."""
    console = _capture_console()
    observer = RichProgressObserver(
        console,
        stages=frozenset({"bw-tcp", "fortio-sat"}),
    )
    with observer:
        observer.on_benchmark_start(1)
        task = _task_for(observer, observer._stage_task_id)
        assert task.total == 2
        observer.on_iteration_start(0)
        for stage in ("bw-tcp", "fortio-sat"):
            observer.on_stage_start(stage, 0)
            observer.on_stage_end(stage, 0)
        task = _task_for(observer, observer._stage_task_id)
        assert task.completed == 2
        # A stray extra on_stage_end must not overshoot the smaller total.
        observer.on_stage_start("bw-udp", 0)
        observer.on_stage_end("bw-udp", 0)
        task = _task_for(observer, observer._stage_task_id)
        assert task.completed == 2


def test_rich_observer_stage_bar_advances_per_stage_end() -> None:
    """``on_stage_end`` advances the stage bar; description tracks the label."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_benchmark_start(1)
        observer.on_iteration_start(0)
        for index, stage in enumerate(
            ("bw-tcp", "bw-udp", "fortio-sat", "fortio-fixed"),
        ):
            observer.on_stage_start(stage, 0)
            task = _task_for(observer, observer._stage_task_id)
            assert task.description == f"Stage [{stage}]"
            observer.on_stage_end(stage, 0)
            task = _task_for(observer, observer._stage_task_id)
            assert task.completed == index + 1


def test_rich_observer_stage_bar_resets_per_iteration() -> None:
    """``on_iteration_start`` zeros the stage bar back to 0/4."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_benchmark_start(2)
        observer.on_iteration_start(0)
        for stage in ("bw-tcp", "bw-udp", "fortio-sat", "fortio-fixed"):
            observer.on_stage_start(stage, 0)
            observer.on_stage_end(stage, 0)
        observer.on_iteration_end(0)
        observer.on_iteration_start(1)
        task = _task_for(observer, observer._stage_task_id)
        assert task.completed == 0
        assert task.description == "Stage"
        # Partial iteration: 2/4 after two stage ends.
        for stage in ("bw-tcp", "bw-udp"):
            observer.on_stage_start(stage, 1)
            observer.on_stage_end(stage, 1)
        task = _task_for(observer, observer._stage_task_id)
        assert task.completed == 2


def test_rich_observer_stage_eta_accumulates_across_iterations_and_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage ETA history carries across iterations and across trial boundaries."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)

    def _do_stage(stage: str, iteration: int, duration: float) -> None:
        observer.on_stage_start(stage, iteration)
        clock["t"] += duration
        observer.on_stage_end(stage, iteration)

    def _do_iter(iteration: int) -> None:
        observer.on_iteration_start(iteration)
        for stage in ("bw-tcp", "bw-udp", "fortio-sat", "fortio-fixed"):
            _do_stage(stage, iteration, 30.0)
        observer.on_iteration_end(iteration)

    with observer:
        # Trial 1: 1 iteration (4 stages at 30s each).
        observer.on_benchmark_start(2)
        _do_iter(0)
        # Trial 2: open iteration 0 so the stage bar is ready for ETA rendering.
        observer.on_benchmark_start(2)
        observer.on_iteration_start(0)
        assert observer._stage_count == 4
        assert observer._stage_total_seconds == pytest.approx(120.0)
        column = _find_eta_column(observer)
        stage_task = _task_for(observer, observer._stage_task_id)
        # 4 remaining stages in the fresh iteration * 30s mean = 0:02:00.
        assert column.render(stage_task).plain == "0:02:00"


def test_rich_observer_aborted_stage_does_not_pollute_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A started-but-never-ended stage is dropped at the iteration boundary."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    clock = {"t": 0.0}

    def _fake() -> float:
        return clock["t"]

    monkeypatch.setattr(observer._progress, "get_time", _fake)
    monkeypatch.setattr(progress_module.time, "monotonic", _fake)
    with observer:
        observer.on_benchmark_start(1)
        observer.on_iteration_start(0)
        observer.on_stage_start("bw-tcp", 0)
        clock["t"] += 10.0
        # Runner raised before the inner try/finally could call on_stage_end;
        # the outer try/finally still fires on_iteration_end.
        observer.on_iteration_end(0)
    assert observer._stage_count == 0
    assert observer._stage_total_seconds == pytest.approx(0.0)
    assert observer._stage_start is None


def test_rich_observer_stage_bar_survives_trial_boundary() -> None:
    """Stage history carries across trials while the bar resets to 0/4."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_benchmark_start(1)
        observer.on_iteration_start(0)
        for stage in ("bw-tcp", "bw-udp", "fortio-sat", "fortio-fixed"):
            observer.on_stage_start(stage, 0)
            observer.on_stage_end(stage, 0)
        observer.on_iteration_end(0)
        assert observer._stage_count == 4
        # Trial 2 begins.
        observer.on_benchmark_start(3)
        task = _task_for(observer, observer._stage_task_id)
        assert task.total == _STAGES_PER_ITERATION
        assert task.completed == 0
        assert task.description == "Stage"
        # History preserved across the trial boundary.
        assert observer._stage_count == 4


def test_rich_observer_iteration_description_stays_current_across_stages() -> None:
    """``on_stage_start`` leaves the iteration bar's description untouched."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_benchmark_start(1)
        observer.on_iteration_start(0)
        observer.on_stage_start("bw-tcp", 0)
        observer.on_stage_end("bw-tcp", 0)
        observer.on_stage_start("bw-udp", 0)
        iter_task = _task_for(observer, observer._iter_task_id)
        stage_task = _task_for(observer, observer._stage_task_id)
        assert iter_task.description == "Current"
        assert stage_task.description == "Stage [bw-udp]"


def test_rich_observer_render_order_is_trials_iteration_stage() -> None:
    """Rich renders the bars top-to-bottom as trials -> iteration -> stage."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_trial_start(0, 3, "sobol", {"net.core.rmem_max": "1048576"})
        observer.on_benchmark_start(2)
        descriptions = [t.description for t in observer._progress.tasks]
        assert descriptions == ["Trials [sobol]", "Current", "Stage"]


def test_rich_observer_stage_bar_advance_clamped_at_total() -> None:
    """A stray fifth ``on_stage_end`` cannot drive the bar past 4/4."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    with observer:
        observer.on_benchmark_start(1)
        observer.on_iteration_start(0)
        for stage in ("bw-tcp", "bw-udp", "fortio-sat", "fortio-fixed"):
            observer.on_stage_start(stage, 0)
            observer.on_stage_end(stage, 0)
        # Simulate a hook-pairing bug: a fifth end call in the same iteration.
        observer.on_stage_end("fortio-fixed", 0)
        task = _task_for(observer, observer._stage_task_id)
        assert task.completed == _STAGES_PER_ITERATION


def test_make_observer_enabled_returns_rich_implementation() -> None:
    console = _capture_console()
    obs = make_observer(enabled=True, console=console)
    assert isinstance(obs, RichProgressObserver)


def test_make_observer_disabled_returns_null() -> None:
    console = _capture_console()
    obs = make_observer(enabled=False, console=console)
    assert isinstance(obs, NullObserver)


# --- score-based Best-so-far ranking --------------------------------------


def _score_row_complete(  # noqa: PLR0913 - keyword-only metric bundle
    observer: RichProgressObserver,
    *,
    index: int,
    phase: str = "bayesian",
    throughput: float,
    tcp_retransmit_rate: float = 0.01,
    udp_jitter: float = 0.1,
    rps: float = 1000.0,
    latency_p50: float = 1.0,
    latency_p90: float = 2.0,
    latency_p99: float = 4.0,
) -> None:
    """Drive ``observer.on_trial_complete`` with a full metric bundle."""
    observer.on_trial_complete(
        index,
        _stub_trial(phase, trial_id=f"trial-{index:02d}"),
        {
            "tcp_throughput": (throughput, 0.0),
            "tcp_retransmit_rate": (tcp_retransmit_rate, 0.0),
            "udp_jitter": (udp_jitter, 0.0),
            "rps": (rps, 0.0),
            "latency_p50": (latency_p50, 0.0),
            "latency_p90": (latency_p90, 0.0),
            "latency_p99": (latency_p99, 0.0),
        },
    )


def test_top5_ranks_by_score_when_objectives_provided() -> None:
    """Two trials: A +1% throughput, B +5% RPS with lower latency -> B wins.

    Under the default objectives and weights, RPS (maximize,
    unweighted) and latency (minimize, weighted) outpull a 1%
    throughput advantage. The throughput-only fallback would put A
    first; the scored path must put B first.
    """
    console = _capture_console()
    observer = RichProgressObserver(console, objectives=ObjectivesSection())
    _score_row_complete(
        observer,
        index=0,
        throughput=1.01e9,
        rps=1000.0,
        latency_p90=5.0,
        latency_p99=10.0,
    )
    _score_row_complete(
        observer,
        index=1,
        throughput=1.00e9,
        rps=1050.0,
        latency_p90=2.0,
        latency_p99=4.0,
    )
    assert [row.index for row in observer._top] == [1, 0]


def test_top5_holds_every_completed_trial_for_rerank() -> None:
    """All rows stay in ``_all_rows`` so the top-5 tracks true rank.

    Append ten monotonically improving trials one at a time; after
    each completion, the ``_top`` must reflect the best five across
    every row seen so far. Guards against a bug where ``_all_rows``
    is pruned alongside ``_top`` (freezing the table on the first
    five arrivals).
    """
    console = _capture_console()
    observer = RichProgressObserver(console, objectives=ObjectivesSection())
    for i in range(10):
        _score_row_complete(
            observer,
            index=i,
            throughput=(1.0 + 0.01 * i) * 1e9,
            latency_p99=10.0 - i,
        )
        assert len(observer._all_rows) == i + 1
        expected_top = sorted(range(i + 1), key=operator.neg)[:5]
        assert [row.index for row in observer._top] == expected_top


def test_top5_phase_labels_roundtrip_through_rerank() -> None:
    """A row's ``phase`` string survives reordering on each refresh.

    Append sobol rows first, then bayesian rows. After the scored
    rerank, every row in ``_top`` must still carry the phase it was
    appended under. Prevents silent label corruption during sort.
    """
    console = _capture_console()
    observer = RichProgressObserver(console, objectives=ObjectivesSection())
    for i in range(3):
        _score_row_complete(
            observer,
            index=i,
            phase="sobol",
            throughput=(1.0 + 0.01 * i) * 1e9,
        )
    for i in range(3, 6):
        _score_row_complete(
            observer,
            index=i,
            phase="bayesian",
            throughput=(1.0 + 0.1 * i) * 1e9,
        )
    by_index = {row.index: row.phase for row in observer._top}
    for idx, phase in by_index.items():
        assert phase == ("sobol" if idx < 3 else "bayesian")


def test_top5_handles_nan_metrics() -> None:
    """A UDP-only trial (NaN retx_per_gb, NaN latency) still ranks."""
    console = _capture_console()
    observer = RichProgressObserver(console, objectives=ObjectivesSection())
    # TCP trial with a full bundle.
    _score_row_complete(
        observer,
        index=0,
        throughput=1.5e9,
        tcp_retransmit_rate=0.01,
        latency_p90=2.0,
        latency_p99=4.0,
    )
    # UDP-only trial: tcp_retransmit_rate and every latency percentile NaN.
    observer.on_trial_complete(
        1,
        _stub_trial("bayesian"),
        {
            "tcp_throughput": (2.0e9, 0.0),
            "tcp_retransmit_rate": (math.nan, math.nan),
            "rps": (math.nan, math.nan),
            "latency_p50": (math.nan, math.nan),
            "latency_p90": (math.nan, math.nan),
            "latency_p99": (math.nan, math.nan),
        },
    )
    top_indices = [row.index for row in observer._top]
    assert set(top_indices) == {0, 1}


def test_top5_fallback_sorts_by_throughput_when_objectives_none() -> None:
    """Bare construction falls back to throughput-descending."""
    console = _capture_console()
    observer = RichProgressObserver(console)
    assert observer._objectives is None
    with observer:
        for i, tp in enumerate([1, 9, 5, 7, 3, 8, 2]):
            observer.on_trial_start(i, 10, "bayesian", {})
            observer.on_trial_complete(
                i,
                _stub_trial("bayesian"),
                {
                    "tcp_throughput": (tp * 1e9, 0.0),
                    "tcp_retransmit_rate": (0.0, 0.0),
                },
            )
    top_mbps = [int(r.throughput_mbps / 1000) for r in observer._top]
    assert top_mbps == [9, 8, 7, 5, 3]
    output = cast("io.StringIO", console.file).getvalue()
    assert "by tcp_throughput" in output
    assert "by weighted score" not in output


def test_best_so_far_hides_disabled_stage_columns() -> None:
    """Columns whose metric is only produced by a disabled stage vanish.

    A bw-tcp-only run renders `Throughput` and `retx/GB`, but must
    drop `jitter`, `RPS`, and `p99` -- those come from bw-udp,
    fortio-sat, and fortio-fixed respectively.
    """
    console = _capture_console()
    observer = RichProgressObserver(console, stages=frozenset({"bw-tcp"}))
    with observer:
        observer.on_trial_start(0, 1, "bayesian", {})
        observer.on_trial_complete(
            0,
            _stub_trial("bayesian"),
            {
                "tcp_throughput": (1.5e9, 0.0),
                "tcp_retransmit_rate": (0.01, 0.0),
                "udp_jitter": (math.nan, math.nan),
                "rps": (math.nan, math.nan),
                "latency_p99": (math.nan, math.nan),
            },
        )
    output = cast("io.StringIO", console.file).getvalue()
    assert "Throughput" in output
    assert "retx/GB" in output
    assert "jitter" not in output
    assert " RPS " not in output
    assert "p99" not in output


def test_best_so_far_shows_all_columns_by_default() -> None:
    """Bare construction (stages=None) keeps every column visible."""
    console = _capture_console()
    observer = RichProgressObserver(console, objectives=ObjectivesSection())
    with observer:
        observer.on_trial_start(0, 1, "bayesian", {})
        _score_row_complete(observer, index=0, throughput=1.0e9)
    output = cast("io.StringIO", console.file).getvalue()
    for header in ("Throughput", "retx/GB", "jitter", "RPS", "p99"):
        assert header in output


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
