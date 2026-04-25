"""Progress-reporting abstractions for long-running CLI commands.

This module provides a single :class:`ProgressObserver` ``Protocol``
that :class:`kube_autotuner.optimizer.OptimizationLoop` and
:class:`kube_autotuner.benchmark.runner.BenchmarkRunner` call into as
their work progresses, plus two implementations: :class:`NullObserver`
(silent; the default) and :class:`RichProgressObserver` (a live
``rich`` display with stacked progress bars and a rolling top-results
table).

The split keeps the domain modules free of ``rich`` imports and lets
library consumers or tests pass :class:`NullObserver` to suppress
output entirely.

Import constraint: this module is reachable from ``cli.py`` at import
time (``task completions`` eagerly imports the Typer ``app``), so it
MUST NOT import ``ax-platform``, ``pandas``, ``plotly``,
``scikit-learn``, or anything under ``kube_autotuner.optimizer`` /
``kube_autotuner.report``. Keep the
import set to ``rich``, the standard library, and pure-typing helpers.
:mod:`kube_autotuner.scoring` is pure-stdlib and :mod:`kube_autotuner.experiment`
is Pydantic-only, so both are safe to import here.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time
from typing import TYPE_CHECKING, Protocol, Self

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table
from rich.text import Text

from kube_autotuner.models import ALL_STAGES, metrics_for_stages
from kube_autotuner.scoring import (
    METRIC_TO_DF_COLUMN,
    config_memory_cost,
    score_rows,
)
from kube_autotuner.sysctl.params import PARAM_SPACE
from kube_autotuner.units import format_coefficient, pick_duration_unit_for_series

try:
    import termios
    import tty
except ImportError:  # Windows
    termios = None  # ty: ignore[invalid-assignment]
    tty = None  # ty: ignore[invalid-assignment]

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType
    from typing import Any

    from rich.console import Console
    from rich.progress import Task, TaskID

    from kube_autotuner.experiment import ObjectivesSection
    from kube_autotuner.models import StageName, TrialResult


_TOP_N = 5
# Default stage count tracks :data:`~kube_autotuner.models.ALL_STAGES`.
# Callers that adopt ``BenchmarkConfig.stages`` override this via
# ``make_observer(stages=...)``.
_STAGES_PER_ITERATION = len(ALL_STAGES)


class ProgressObserver(Protocol):
    """Progress callbacks emitted by the optimize / benchmark loops.

    Every method is fire-and-forget: implementations must not raise
    and must tolerate out-of-order or missing calls.

    One ordering invariant is guaranteed: inside a single benchmark
    run, ``on_benchmark_start`` fires once before any
    ``on_iteration_start`` from that run. This lets implementations
    size a trial-wide iteration progress bar up front.

    Implementations must also be usable as context managers; the CLI
    wraps the entire run in ``with observer:`` so live displays have a
    defined lifecycle.
    """

    def __enter__(self) -> Self:
        """Start any live display owned by the observer."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Tear down the live display, flushing any pending output."""
        ...

    def seed_history(self, prior: list[TrialResult], n_sobol: int) -> None:
        """Pre-populate the observer with prior trial results.

        Called once before the optimize loop begins when resuming from
        a previous session's trial dataset. The default implementation is a
        no-op; :class:`RichProgressObserver` overrides it to rebuild
        the "best so far" table so the live display does not start
        blank after a resume.

        Args:
            prior: Successful :class:`TrialResult` records from the
                prior session, in file order.
            n_sobol: The Sobol initialization budget for the
                experiment; used to pick the live progress bar's phase
                label on resume.
        """
        ...

    def on_trial_start(
        self,
        index: int,
        total: int,
        phase: str,
        params: Mapping[str, str],
    ) -> None:
        """Report that trial ``index`` of ``total`` is starting.

        Args:
            index: Zero-based trial index.
            total: Total trials planned for this run.
            phase: ``"sobol"`` during Ax initialization, ``"bayesian"``
                afterwards.
            params: The Ax-proposed sysctl parameterization about to
                be applied.
        """
        ...

    def on_trial_complete(
        self,
        index: int,
        trial: TrialResult,
        metrics: Mapping[str, tuple[float, float]],
    ) -> None:
        """Report that trial ``index`` finished successfully.

        Args:
            index: Zero-based trial index.
            trial: The :class:`TrialResult` just persisted. Carries
                ``phase``, ``trial_id``, and ``parent_trial_id`` so the
                observer can key its internal aggregation and render
                correctly for both primary and verification rows.
            metrics: The per-metric ``(mean, SEM)`` table returned by
                the optimizer's metric aggregation.
        """
        ...

    def on_verification_start(self, top_k: int, total_remaining: int) -> None:
        """Announce the start of the post-primary verification phase.

        Called once by
        :meth:`kube_autotuner.optimizer.OptimizationLoop.run_verification`
        before the first verification ``on_trial_start``. Rich-backed
        observers use ``total_remaining`` to size a dedicated
        verification progress bar; ``total_remaining`` already reflects
        work skipped by a resume (it is
        ``sum(max(0, repeats - already_done_by_parent.get(p, 0))
        for p in parents)``), so the bar does not overshoot on partial
        resumes. There is no matching ``on_verification_end`` -- the
        existing ``__exit__`` handles teardown.

        Args:
            top_k: Number of parent configs selected for verification.
            total_remaining: Total verification runs yet to execute in
                the current session (post-``already_done_by_parent``).
        """
        ...

    def on_trial_failed(self, index: int, exc: BaseException) -> None:
        """Report that trial ``index`` raised before completing.

        Args:
            index: Zero-based trial index.
            exc: The exception that aborted the trial.
        """
        ...

    def on_benchmark_start(self, total_iterations: int) -> None:
        """Report that a benchmark run is starting.

        Fires once per :meth:`BenchmarkRunner.run` call, before any
        ``on_iteration_start``. Defines the trial-wide iteration
        budget so implementations can size their iteration progress
        bar up front. Implementations must preserve any cumulative
        iteration history (mean-duration samples used to compute ETA)
        across this call -- only the current benchmark's
        progress-bar state should be reset.

        Args:
            total_iterations: Total iterations planned for the
                benchmark run, i.e. ``config.iterations``.
        """
        ...

    def on_iteration_start(self, iteration: int) -> None:
        """Report that a single benchmark iteration is starting.

        Args:
            iteration: Zero-based iteration index.
        """
        ...

    def on_iteration_end(self, iteration: int) -> None:
        """Report that a single benchmark iteration has finished.

        Args:
            iteration: Zero-based iteration index.
        """
        ...

    def on_stage_start(self, stage: str, iteration: int) -> None:
        """Report that a sub-stage within an iteration is starting.

        Each iteration expands into four sub-stages: ``"bw-tcp"``
        (iperf3 TCP), ``"bw-udp"`` (iperf3 UDP), ``"fortio-sat"``
        (fortio ``-qps 0``), and ``"fortio-fixed"`` (fortio at the
        configured fixed QPS). Observers can use the label to show
        which sub-stage is live.

        Args:
            stage: Sub-stage label.
            iteration: Zero-based iteration index.
        """
        ...

    def on_stage_end(self, stage: str, iteration: int) -> None:
        """Report that a sub-stage within an iteration has finished.

        Args:
            stage: Sub-stage label.
            iteration: Zero-based iteration index.
        """
        ...


class NullObserver:
    """No-op :class:`ProgressObserver` used when progress UI is off."""

    def __enter__(self) -> Self:
        """Return ``self`` unchanged; there is no live display to start.

        Returns:
            The observer instance, as required by the context-manager
            protocol.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """No-op."""

    def seed_history(self, prior: list[TrialResult], n_sobol: int) -> None:
        """No-op."""

    def on_trial_start(
        self,
        index: int,
        total: int,
        phase: str,
        params: Mapping[str, str],
    ) -> None:
        """No-op."""

    def on_trial_complete(
        self,
        index: int,
        trial: TrialResult,
        metrics: Mapping[str, tuple[float, float]],
    ) -> None:
        """No-op."""

    def on_verification_start(self, top_k: int, total_remaining: int) -> None:
        """No-op."""

    def on_trial_failed(self, index: int, exc: BaseException) -> None:
        """No-op."""

    def on_benchmark_start(self, total_iterations: int) -> None:
        """No-op."""

    def on_iteration_start(self, iteration: int) -> None:
        """No-op."""

    def on_iteration_end(self, iteration: int) -> None:
        """No-op."""

    def on_stage_start(self, stage: str, iteration: int) -> None:
        """No-op."""

    def on_stage_end(self, stage: str, iteration: int) -> None:
        """No-op."""


class _HistoryEtaColumn(ProgressColumn):
    """ETA column backed by an observer-owned duration history.

    The same column powers two progress bars for two reasons that both
    rule out Rich's stock ``TimeRemainingColumn`` (which reads
    ``Task.speed``):

    1. *Iteration bar.* ``Task.speed`` is backed by a sample deque that
       ``Progress.reset()`` clears. The iteration bar used to be reset
       on every mode boundary to restore the per-mode ``x/N`` display,
       so those samples vanished at every mode and trial boundary even
       after ``speed_estimate_period`` pruning was disabled.
    2. *Trial bar.* ``Task.speed`` requires **at least two** progress
       samples to produce a value (independent of pruning). With only
       one ``advance=1`` call posted after the first completion, the
       column would render ``-:--:--`` until the second trial finished
       -- roughly 19 minutes into a 9.5-min-per-trial run.

    Both bars supply a ``(count, total_seconds)`` snapshot through a
    caller-provided callable, letting the ETA be available after the
    very first completion and span the entire run.
    """

    def __init__(
        self,
        history: Callable[[TaskID], tuple[int, float, float]],
    ) -> None:
        """Bind the column to a per-task ``(count, total_seconds, in_flight)`` supplier.

        Args:
            history: One-arg callable taking the Rich ``TaskID`` and
                returning the observer's cumulative completed count,
                total observed duration in seconds, and the elapsed
                time of the currently in-flight unit (``0.0`` when no
                unit is active). The in-flight component lets Rich's
                8 Hz refresh tick the ETA down second-by-second within
                a unit instead of holding flat until the next
                completion. Dispatching by task id lets a single
                column serve every task in a shared ``Progress``
                (trial bar, iteration bar, verify bar).
        """
        super().__init__()
        self._history = history

    def render(self, task: Task) -> Text:
        """Render the ETA as ``H:MM:SS`` or ``-:--:--`` when unknown.

        Args:
            task: The Rich task whose remaining work is being measured.

        Returns:
            A styled :class:`rich.text.Text` carrying the ETA string.
        """
        if task.total is None:
            return Text("-:--:--", style="progress.remaining")
        remaining = int(task.total) - int(task.completed)
        count, total_seconds, in_flight = self._history(task.id)
        if count == 0 or remaining <= 0:
            return Text("-:--:--", style="progress.remaining")
        mean = total_seconds / count
        # (remaining - 1) future units at the mean, plus whatever is
        # left of the in-flight one. Floors at 0 when the in-flight
        # unit has already exceeded the mean; parks the ETA on
        # (remaining - 1) * mean until it completes and the mean
        # recalculates.
        eta_seconds = int((remaining - 1) * mean + max(0.0, mean - in_flight))
        minutes, seconds = divmod(eta_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return Text(
            f"{hours:d}:{minutes:02d}:{seconds:02d}",
            style="progress.remaining",
        )


class _TrialRow:
    """One row of the live "best so far" table."""

    __slots__ = (
        "index",
        "jitter_seconds",
        "memory_cost",
        "metrics",
        "p99_seconds",
        "parent_trial_id",
        "phase",
        "retx_per_gb",
        "rps",
        "throughput_mbps",
        "trial_id",
    )

    def __init__(  # noqa: PLR0913 keyword-only summary bundle for one trial row
        self,
        *,
        index: int,
        phase: str,
        trial_id: str,
        parent_trial_id: str | None,
        throughput_mbps: float,
        retx_per_gb: float,
        jitter_seconds: float,
        rps: float,
        p99_seconds: float,
        metrics: dict[str, float],
        memory_cost: float,
    ) -> None:
        """Store one trial's summary for display and scoring.

        Args:
            index: Zero-based trial index within this run.
            phase: ``"sobol"``, ``"bayesian"``, or ``"verification"``.
            trial_id: The trial's stable ``TrialResult.trial_id``.
                Primary rows key their own aggregation group by this
                value; verification rows link to the parent via
                ``parent_trial_id``.
            parent_trial_id: The primary trial's ``trial_id`` when
                ``phase == "verification"``; ``None`` otherwise.
            throughput_mbps: Mean throughput in megabits-per-second.
            retx_per_gb: Retransmits per gigabyte; ``NaN`` when the
                trial produced no TCP bytes (every ``bw-tcp`` stage
                failed).
            jitter_seconds: Mean UDP jitter in seconds from the
                ``bw-udp`` stage; ``NaN`` when no UDP record
                reported jitter. Stored in native seconds so the
                live-table unit can be picked at render time.
            rps: Mean requests-per-second from the fortio saturation
                sub-stage; ``NaN`` when the trial produced no
                saturation records.
            p99_seconds: Fixed-QPS p99 latency in seconds; ``NaN``
                when the trial produced no fortio records. Stored
                in native seconds so the live-table unit can be
                picked at render time.
            metrics: Raw-domain per-metric means keyed by the
                :data:`kube_autotuner.scoring.METRIC_TO_DF_COLUMN`
                DataFrame column name (``mean_tcp_throughput``,
                ``mean_udp_throughput``, ...). Fed to
                :func:`kube_autotuner.scoring.score_rows` for the
                live top-N ranking, which min-max normalizes across
                every stored row on each refresh. Stores ``NaN`` for
                metrics the trial did not observe.
            memory_cost: Estimated kernel/CNI memory footprint in
                bytes, precomputed via
                :func:`kube_autotuner.scoring.config_memory_cost` off
                ``TrialResult.sysctl_values``. Cached here so the
                rerank hot path does not re-evaluate the per-param
                cost rules on every refresh.
        """
        self.index = index
        self.phase = phase
        self.trial_id = trial_id
        self.parent_trial_id = parent_trial_id
        self.throughput_mbps = throughput_mbps
        self.retx_per_gb = retx_per_gb
        self.jitter_seconds = jitter_seconds
        self.rps = rps
        self.p99_seconds = p99_seconds
        self.metrics = metrics
        self.memory_cost = memory_cost


def _build_trial_row(
    index: int,
    phase: str,
    metrics: Mapping[str, tuple[float, float]],
    *,
    trial_id: str,
    parent_trial_id: str | None,
    memory_cost: float,
) -> _TrialRow:
    """Project a completed-trial metric bundle into a :class:`_TrialRow`.

    Handles two concerns in one pass:

    * derives the display-domain floats (Mbps / retx/GB) for the
      rendered table, and
    * captures the full raw-domain metric bundle keyed by
      :data:`kube_autotuner.scoring.METRIC_TO_DF_COLUMN` so
      :func:`kube_autotuner.scoring.score_rows` can min-max normalize
      across every metric the user's objectives reference.

    Args:
        index: Zero-based trial index within this run.
        phase: ``"sobol"``, ``"bayesian"``, or ``"verification"``.
        metrics: The ``(mean, SEM)`` bundle from the optimizer.
        trial_id: The trial's stable ``TrialResult.trial_id``.
        parent_trial_id: The primary trial's ``trial_id`` when
            ``phase == "verification"``; ``None`` otherwise.
        memory_cost: Cached static memory footprint in bytes (see
            :func:`kube_autotuner.scoring.config_memory_cost`).

    Returns:
        A :class:`_TrialRow` ready to append to ``_all_rows``.
    """
    tp_pair = metrics.get("tcp_throughput")
    rate_pair = metrics.get("tcp_retransmit_rate")
    jitter_pair = metrics.get("udp_jitter")
    rps_pair = metrics.get("rps")
    p99_pair = metrics.get("latency_p99")
    tp = (tp_pair[0] if tp_pair is not None else 0.0) / 1e6
    rate = rate_pair[0] if rate_pair is not None else math.nan
    retx_per_gb = math.nan if math.isnan(rate) else rate
    jitter_seconds = jitter_pair[0] if jitter_pair is not None else math.nan
    rps = rps_pair[0] if rps_pair is not None else math.nan
    p99_seconds = p99_pair[0] if p99_pair is not None else math.nan
    raw_metrics: dict[str, float] = {
        METRIC_TO_DF_COLUMN[key]: (pair[0] if pair is not None else math.nan)
        for key, pair in (
            ("tcp_throughput", tp_pair),
            ("udp_throughput", metrics.get("udp_throughput")),
            ("tcp_retransmit_rate", rate_pair),
            ("udp_loss_rate", metrics.get("udp_loss_rate")),
            ("udp_jitter", jitter_pair),
            ("rps", rps_pair),
            ("latency_p50", metrics.get("latency_p50")),
            ("latency_p90", metrics.get("latency_p90")),
            ("latency_p99", p99_pair),
        )
    }
    return _TrialRow(
        index=index,
        phase=phase,
        trial_id=trial_id,
        parent_trial_id=parent_trial_id,
        throughput_mbps=tp,
        retx_per_gb=retx_per_gb,
        jitter_seconds=jitter_seconds,
        rps=rps,
        p99_seconds=p99_seconds,
        metrics=raw_metrics,
        memory_cost=memory_cost,
    )


class _TtyEchoSuppressor:
    """Disable stdin ECHO/ICANON for the lifetime of a ``with`` block.

    Reuses :func:`tty.setcbreak`, which clears ``ECHO | ICANON`` while
    preserving ``ISIG`` (so Ctrl-C still raises SIGINT) and returns the
    previously saved termios mode for restoration. Without this, the
    tty driver echoes arrow-key escape sequences directly under a
    ``rich.live.Live`` region, desyncing Rich's cursor tracking and
    stranding a duplicate render in the scrollback on every keypress.

    No-ops silently on Windows (``termios``/``tty`` import failed),
    when stdin is detached or closed, when stdin is not a tty (piped
    input, CliRunner capture), or when the termios calls fail.
    """

    def __init__(self) -> None:
        """Start disarmed; ``__enter__`` attempts the suppression."""
        self._fd: int | None = None
        self._saved: list[Any] | None = None

    def __enter__(self) -> Self:
        """Switch stdin to cbreak mode if possible; record prior state.

        Returns:
            The suppressor instance so callers can use
            ``with _TtyEchoSuppressor() as s:``.
        """
        if termios is None or tty is None:
            return self
        try:
            fd = sys.stdin.fileno()
        except AttributeError, ValueError, OSError:
            return self
        if not os.isatty(fd):
            return self
        try:
            # setcbreak returns the prior termios attrs in 3.12+.
            # TCSAFLUSH (positional) discards already-buffered
            # keystrokes so escape bytes typed moments before we
            # entered don't echo after the mode switch.
            saved = tty.setcbreak(fd, termios.TCSAFLUSH)
        except OSError, termios.error:
            return self
        self._fd = fd
        self._saved = saved
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Restore the original termios attributes if we changed them."""
        del exc_type, exc, tb
        if self._fd is None or self._saved is None or termios is None:
            return
        with contextlib.suppress(OSError, termios.error):
            termios.tcsetattr(self._fd, termios.TCSAFLUSH, self._saved)


class RichProgressObserver:
    """Live ``rich`` display with a trial bar, iteration bar, and results table.

    The observer must be entered as a context manager. It is safe to
    re-enter sequentially (construct, enter, exit, enter again) but
    not nested.
    """

    def __init__(
        self,
        console: Console,
        objectives: ObjectivesSection | None = None,
        stages: frozenset[StageName] | None = None,
    ) -> None:
        """Bind the observer to a shared ``rich`` console.

        Args:
            console: Console to render through. Must be the same
                instance passed to any ``RichHandler`` attached to
                the root logger so log records interleave above the
                live region instead of tearing through it.
            objectives: Pareto objectives and recommendation weights
                from the active
                :class:`~kube_autotuner.experiment.ExperimentConfig`.
                When set, the ``Best so far`` panel ranks every
                completed trial through
                :func:`kube_autotuner.scoring.score_rows` so the
                live view matches the post-hoc recommendation output.
                When ``None`` (bare construction, e.g. unit tests or
                non-optimize flows), the panel falls back to a
                throughput-descending sort.
            stages: Benchmark sub-stages the runner will execute
                each iteration. Sizes the Stage bar so it always
                reaches ``100%`` when the full configured stage set
                completes, and filters the ``Best so far`` table so
                columns whose metrics are produced only by disabled
                stages are hidden. Defaults to
                :data:`~kube_autotuner.models.ALL_STAGES` so callers
                that have not adopted :attr:`BenchmarkConfig.stages`
                keep the same UI.
        """
        self._console = console
        self._objectives = objectives
        self._stages: frozenset[StageName] = (
            stages if stages is not None else ALL_STAGES
        )
        self._stages_per_iteration = len(self._stages)
        self._stage_metrics: frozenset[str] = metrics_for_stages(self._stages)
        self._iter_start: float | None = None
        self._iter_count: int = 0
        self._iter_total_seconds: float = 0.0
        self._stage_start: float | None = None
        self._stage_count: int = 0
        self._stage_total_seconds: float = 0.0
        self._trial_start: float | None = None
        self._trial_count: int = 0
        self._trial_total_seconds: float = 0.0
        self._trial_task_id: TaskID | None = None
        self._iter_task_id: TaskID | None = None
        self._stage_task_id: TaskID | None = None
        self._verify_task_id: TaskID | None = None
        self._progress: Progress = Progress(
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(table_column=Column(justify="right", no_wrap=True)),
            TextColumn("[dim]elapsed[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/dim]"),
            _HistoryEtaColumn(self._history_for_task),
            expand=True,
        )
        self._all_rows: list[_TrialRow] = []
        self._top: list[_TrialRow] = []
        self._phase: str = "sobol"
        self._live: Live | None = None
        self._echo: _TtyEchoSuppressor | None = None

    def _iter_history(self) -> tuple[int, float, float]:
        """Return the observer's cumulative iteration history.

        Returns:
            A ``(count, total_seconds, in_flight_seconds)`` snapshot
            aggregated across every completed iteration since the
            observer was constructed. ``in_flight_seconds`` is the
            elapsed time of the currently running iteration (``0.0``
            when none is active), so the ETA column can tick the
            estimate down every refresh.
        """
        in_flight = (
            time.monotonic() - self._iter_start if self._iter_start is not None else 0.0
        )
        return (self._iter_count, self._iter_total_seconds, in_flight)

    def _stage_history(self) -> tuple[int, float, float]:
        """Return the observer's cumulative sub-stage history.

        Stages average tens of seconds while iterations average
        minutes, so the stage bar keeps its own history rather than
        sharing :meth:`_iter_history` -- reusing iteration samples
        would overshoot the stage ETA by ~4x.

        Returns:
            A ``(count, total_seconds, in_flight_seconds)`` snapshot
            aggregated across every completed sub-stage since the
            observer was constructed. ``in_flight_seconds`` is the
            elapsed time of the currently running stage (``0.0`` when
            none is active).
        """
        in_flight = (
            time.monotonic() - self._stage_start
            if self._stage_start is not None
            else 0.0
        )
        return (self._stage_count, self._stage_total_seconds, in_flight)

    def _trial_history(self) -> tuple[int, float, float]:
        """Return the observer's cumulative trial history.

        Returns:
            A ``(count, total_seconds, in_flight_seconds)`` snapshot
            aggregated across every completed or failed trial since
            the observer was constructed. ``in_flight_seconds`` is the
            elapsed time of the currently running trial (``0.0`` when
            between trials). Failed trials contribute because they
            still consumed wall time; omitting them would
            under-estimate the ETA on runs with failures.
        """
        in_flight = (
            time.monotonic() - self._trial_start
            if self._trial_start is not None
            else 0.0
        )
        return (self._trial_count, self._trial_total_seconds, in_flight)

    def _history_for_task(
        self,
        task_id: TaskID,
    ) -> tuple[int, float, float]:
        """Return the ``(count, total_seconds, in_flight)`` snapshot for ``task_id``.

        The single :class:`_HistoryEtaColumn` routed into
        ``self._progress`` calls this for every task it renders. The
        trial and verify tasks share the observer-owned trial
        history (both measure wall-time per Ax trial), while the
        iteration bar uses its own per-iteration history. Unknown task
        ids fall through to the empty snapshot so ``render`` draws
        ``-:--:--`` instead of raising.

        Args:
            task_id: The Rich ``TaskID`` of the task whose ETA is
                being rendered.

        Returns:
            The matching history tuple, or ``(0, 0.0, 0.0)`` when the
            task id is not one the observer owns.
        """
        if task_id in {self._trial_task_id, self._verify_task_id}:
            return self._trial_history()
        if task_id == self._iter_task_id:
            return self._iter_history()
        if self._stage_task_id is not None and task_id == self._stage_task_id:
            return self._stage_history()
        return (0, 0.0, 0.0)

    def _render(self) -> Group:
        """Compose the live renderable (the shared bars over a table).

        The "best so far" table is omitted until at least one trial
        has completed, so the ``baseline`` and ``trial`` runs (which
        never call :meth:`on_trial_complete`) show bars only rather
        than an empty table frame.

        Returns:
            A :class:`rich.console.Group` of the shared progress
            region (trial, iteration, and verify tasks render
            inside one ``Progress`` so their columns stay aligned)
            and, when any trial has completed, the current "best so
            far" table.
        """
        if not self._top:
            return Group(self._progress)
        sort_key = (
            "weighted score" if self._objectives is not None else "tcp_throughput"
        )
        has_verification = any(r.phase == "verification" for r in self._all_rows)
        suffix = ", verified" if has_verification else ""
        table = Table(
            title=f"Best so far (top {_TOP_N} by {sort_key}{suffix})",
            title_style="bold",
            show_header=True,
            header_style="bold",
            expand=True,
        )
        relevant = self._stage_metrics
        # ``metric=None`` marks identity columns that are always shown.
        # Duration-unit picking is gated on the owning stage so we
        # don't run pick_duration_unit_for_series over an all-NaN series.
        specs: list[tuple[str | None, str, Callable[[_TrialRow], str]]] = [
            (None, "#", lambda r: str(r.index + 1)),
            (None, "Phase", lambda r: r.phase),
            (
                "tcp_throughput",
                "Throughput",
                lambda r: f"{r.throughput_mbps:,.1f} Mbps",
            ),
            (
                "tcp_retransmit_rate",
                "retx/GB",
                lambda r: (
                    "n/a" if math.isnan(r.retx_per_gb) else f"{r.retx_per_gb:.2f}"
                ),
            ),
        ]
        if "udp_jitter" in relevant:
            jitter_scale, jitter_suffix = pick_duration_unit_for_series(
                r.jitter_seconds for r in self._top
            )
            specs.append((
                "udp_jitter",
                f"jitter {jitter_suffix}",
                lambda r: (
                    "n/a"
                    if math.isnan(r.jitter_seconds)
                    else format_coefficient(r.jitter_seconds / jitter_scale)
                ),
            ))
        specs.append((
            "rps",
            "RPS",
            lambda r: "n/a" if math.isnan(r.rps) else f"{r.rps:,.1f}",
        ))
        if "latency_p99" in relevant:
            p99_scale, p99_suffix = pick_duration_unit_for_series(
                r.p99_seconds for r in self._top
            )
            specs.append((
                "latency_p99",
                f"p99 {p99_suffix}",
                lambda r: (
                    "n/a"
                    if math.isnan(r.p99_seconds)
                    else format_coefficient(r.p99_seconds / p99_scale)
                ),
            ))
        active = [s for s in specs if s[0] is None or s[0] in relevant]
        for metric, header, _cell in active:
            justify = "right" if metric is not None or header == "#" else "left"
            no_wrap = header in {"#", "Phase"}
            table.add_column(header, justify=justify, no_wrap=no_wrap)
        for row in self._top:
            table.add_row(*(cell(row) for _metric, _header, cell in active))
        return Group(self._progress, table)

    def _refresh(self) -> None:
        """Stage the current renderable for the next auto-refresh tick."""
        if self._live is not None:
            self._live.update(self._render())

    def _task_by_id(self, task_id: TaskID) -> Task | None:
        """Return the Rich ``Task`` for ``task_id`` or ``None`` if absent.

        Rich does not expose a public dict lookup, so this linear scan
        over ``self._progress.tasks`` (at most four entries -- trial,
        iteration, stage, verify) is the cheapest stable API.

        Args:
            task_id: The Rich ``TaskID`` to resolve.

        Returns:
            The matching task, or ``None`` when no task with the given
            id is registered.
        """
        return next((t for t in self._progress.tasks if t.id == task_id), None)

    def __enter__(self) -> Self:
        """Start the ``rich.live.Live`` display on the bound console.

        Stdin ECHO/ICANON is cleared first via :class:`_TtyEchoSuppressor`
        so keystrokes (notably arrow-key escape sequences) do not echo
        under the live region and desync Rich's cursor tracking.

        Returns:
            The observer instance so callers can use ``with obs as o:``.
        """
        echo = _TtyEchoSuppressor()
        echo.__enter__()
        try:
            # Rely on Rich's default refresh cadence (4 Hz); higher rates
            # strobe on backgrounded macOS terminals whose paint loop is
            # throttled below our write cadence.
            live = Live(
                self._render(),
                console=self._console,
                transient=False,
            )
            live.__enter__()
        except BaseException:
            # BaseException (not Exception) so KeyboardInterrupt /
            # SystemExit during Live construction still unwinds echo
            # suppression -- leaving the terminal in noecho mode would
            # strand the user.
            echo.__exit__(None, None, None)
            raise
        self._echo = echo
        self._live = live
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the live display and restore stdin echo; swallow no exceptions."""
        try:
            if self._live is not None:
                self._live.__exit__(exc_type, exc, tb)
                self._live = None
        finally:
            if self._echo is not None:
                self._echo.__exit__(exc_type, exc, tb)
                self._echo = None

    def on_trial_start(
        self,
        index: int,
        total: int,
        phase: str,
        params: Mapping[str, str],  # noqa: ARG002 reserved for future detail pane
    ) -> None:
        """Create the trial bar on first call, advance the phase label."""
        self._phase = phase
        description = f"Trial [{phase}]"
        if self._trial_task_id is None:
            self._trial_task_id = self._progress.add_task(
                description,
                total=total,
                completed=index,
            )
        else:
            self._progress.update(
                self._trial_task_id,
                description=description,
                total=total,
            )
        self._trial_start = time.monotonic()
        self._refresh()

    def on_trial_complete(
        self,
        index: int,
        trial: TrialResult,
        metrics: Mapping[str, tuple[float, float]],
    ) -> None:
        """Advance the trial bar and fold the result into the top-N table."""
        phase = trial.phase or "bayesian"
        if self._trial_start is not None:
            self._trial_total_seconds += time.monotonic() - self._trial_start
            self._trial_count += 1
            self._trial_start = None
        if phase != "verification" and self._trial_task_id is not None:
            self._progress.update(self._trial_task_id, advance=1)
        if phase == "verification" and self._verify_task_id is not None:
            self._progress.update(self._verify_task_id, advance=1)
        row = _build_trial_row(
            index,
            phase,
            metrics,
            trial_id=trial.trial_id,
            parent_trial_id=trial.parent_trial_id,
            memory_cost=config_memory_cost(trial.sysctl_values, PARAM_SPACE),
        )
        self._all_rows.append(row)
        self._rerank()
        self._refresh()

    def seed_history(self, prior: list[TrialResult], n_sobol: int) -> None:
        """Pre-populate the "best so far" table from prior trials.

        Called once at loop-start when resuming from a prior session.
        Primary rows carry a stored ``phase`` (``"sobol"`` or
        ``"bayesian"``); verification rows carry ``phase="verification"``
        and do not shift the Sobol/Bayesian index over primary rows. The
        two ranges are interleaved in the observer's ``_all_rows`` so
        dataset file order is preserved, and :meth:`_rerank` routes
        through the aggregation path when any verification row is
        present.

        ``_trial_count`` / ``_trial_total_seconds`` are deliberately
        left at zero so the ETA re-learns from live wall-clock; the
        first ``on_trial_start`` advances the Rich bar to
        ``completed=index`` to reflect the seeded position.

        Args:
            prior: Prior successful :class:`TrialResult` records in
                file order.
            n_sobol: The current experiment's Sobol budget; used to
                pick the Rich progress bar's phase label on resume.
        """
        # Lazy imports: ``_compute_metrics`` lives in ``optimizer`` which
        # is reachable only when the ``optimize`` dep group is
        # installed. Keeping the import inside the method preserves
        # the module-level constraint that ``progress`` stays
        # optimize-free for ``task completions``.
        from kube_autotuner.models import is_primary  # noqa: PLC0415
        from kube_autotuner.optimizer import _compute_metrics  # noqa: PLC0415

        primary_count = 0
        for idx, tr in enumerate(prior):
            metrics = _compute_metrics(tr)
            if is_primary(tr):
                primary_count += 1
                assert tr.phase is not None  # noqa: S101 - run_optimize invariant
                phase = tr.phase
            else:
                phase = "verification"
            self._all_rows.append(
                _build_trial_row(
                    idx,
                    phase,
                    metrics,
                    trial_id=tr.trial_id,
                    parent_trial_id=tr.parent_trial_id,
                    memory_cost=config_memory_cost(tr.sysctl_values, PARAM_SPACE),
                ),
            )
        self._rerank()
        if primary_count:
            self._phase = "sobol" if primary_count < n_sobol else "bayesian"

    def _rerank(self) -> None:
        """Refresh ``self._top`` from ``self._all_rows``.

        When an :class:`~kube_autotuner.experiment.ObjectivesSection`
        is bound the ranking goes through
        :func:`kube_autotuner.scoring.score_rows` so the panel
        matches ``recommend_configs``. Without one (bare observer
        construction, ``baseline`` / ``trial`` flows) the
        throughput-descending sort is used.

        Once any ``_all_rows`` entry carries ``phase="verification"``,
        ranking aggregates each parent's primary + verification
        samples first (mirroring
        :func:`kube_autotuner.scoring.aggregate_verification`) and
        keys the rendered top by the parent's stored ``_TrialRow``.
        The pre-verification path stays ungrouped -- hot path, no
        per-refresh allocation change.
        """
        if self._objectives is None:
            self._top = sorted(
                self._all_rows,
                key=lambda r: r.throughput_mbps,
                reverse=True,
            )[:_TOP_N]
            return

        has_verification = any(r.phase == "verification" for r in self._all_rows)
        if has_verification:
            self._top = self._rerank_aggregated()
            return

        scores = score_rows(
            [r.metrics for r in self._all_rows],
            self._objectives.pareto,
            self._objectives.recommendation_weights,
            memory_costs=[r.memory_cost for r in self._all_rows],
            memory_cost_weight=self._objectives.memory_cost_weight,
        )
        # Rank by score desc, break ties by trial_id ascending to
        # match recommend_configs (kube_autotuner.report.analysis) and
        # OptimizationLoop.run_verification's top-K selector.
        order = sorted(
            range(len(self._all_rows)),
            key=lambda i: (-scores[i], self._all_rows[i].trial_id),
        )
        self._top = [self._all_rows[i] for i in order[:_TOP_N]]

    def _rerank_aggregated(self) -> list[_TrialRow]:
        """Return the top rows grouped by parent for verification panels.

        Groups ``_all_rows`` by ``parent_trial_id or trial_id`` (same
        key as :func:`kube_autotuner.scoring.aggregate_verification`),
        means each group's raw metrics into a single row dict, scores
        those, and returns the parent's stored ``_TrialRow`` for each
        winning group so the live panel keeps rendering display-domain
        fields unchanged.

        Returns:
            The top ``_TOP_N`` parent rows ranked by combined score,
            with ties broken by ``trial_id`` ascending.
        """
        # ``_objectives is not None`` is an invariant of the caller --
        # the aggregated branch is only selected in that case.
        assert self._objectives is not None  # noqa: S101 - caller invariant

        groups: dict[str, list[_TrialRow]] = {}
        group_order: list[str] = []
        for row in self._all_rows:
            key = row.parent_trial_id or row.trial_id
            if key not in groups:
                group_order.append(key)
                groups[key] = []
            groups[key].append(row)

        aggregated: list[dict[str, float]] = []
        for key in group_order:
            samples = groups[key]
            agg: dict[str, float] = {}
            for col in METRIC_TO_DF_COLUMN.values():
                values = [
                    r.metrics[col]
                    for r in samples
                    if not math.isnan(r.metrics.get(col, math.nan))
                ]
                agg[col] = math.nan if not values else sum(values) / len(values)
            aggregated.append(agg)

        # Pick each group's "primary row" for display: prefer a
        # non-verification sample (the original primary); fall back to
        # whatever sample is first in the group (orphaned verification
        # rows on a dropped parent).
        display_rows: list[_TrialRow] = []
        for key in group_order:
            samples = groups[key]
            primary = next(
                (r for r in samples if r.phase != "verification"),
                samples[0],
            )
            display_rows.append(primary)

        # Parent and verification repeats share sysctl_values and
        # therefore share memory_cost; pull it off the display (primary)
        # row.
        memory_costs = [r.memory_cost for r in display_rows]

        scores = score_rows(
            aggregated,
            self._objectives.pareto,
            self._objectives.recommendation_weights,
            memory_costs=memory_costs,
            memory_cost_weight=self._objectives.memory_cost_weight,
        )

        order = sorted(
            range(len(group_order)),
            key=lambda i: (-scores[i], display_rows[i].trial_id),
        )
        return [display_rows[i] for i in order[:_TOP_N]]

    def on_verification_start(self, top_k: int, total_remaining: int) -> None:
        """Create a second progress task for the verification phase.

        The primary trial bar is left in place so the live panel keeps
        its completed-primary history visible. ``total_remaining`` has
        already been trimmed by ``already_done_by_parent`` in
        :meth:`OptimizationLoop.run_verification`, so the bar sizes
        correctly on partial resumes and reaches 100% when every
        remaining run has completed.

        Args:
            top_k: Number of parent configs selected for verification.
            total_remaining: Total verification runs yet to execute.
        """
        description = f"Verify [{top_k} configs, {total_remaining} runs]"
        if self._verify_task_id is None:
            self._verify_task_id = self._progress.add_task(
                description,
                total=total_remaining,
                completed=0,
            )
        else:
            self._progress.update(
                self._verify_task_id,
                description=description,
                total=total_remaining,
                completed=0,
            )
        self._refresh()

    def on_trial_failed(
        self,
        index: int,  # noqa: ARG002 recorded for future failure panel
        exc: BaseException,  # noqa: ARG002
    ) -> None:
        """Advance the trial bar without adding a row to the results table."""
        if self._trial_start is not None:
            self._trial_total_seconds += time.monotonic() - self._trial_start
            self._trial_count += 1
            self._trial_start = None
        if self._trial_task_id is not None:
            self._progress.update(self._trial_task_id, advance=1)
        self._refresh()

    def on_benchmark_start(self, total_iterations: int) -> None:
        """Size the iteration and stage bars for the trial's iteration budget.

        Fired once per :meth:`BenchmarkRunner.run` call. The iteration
        bar spans every iteration in the trial (``total=iterations``)
        while the stage bar is sized to the enabled-sub-stage count
        inside each iteration (``total=self._stages_per_iteration``)
        and is re-zeroed on every :meth:`on_iteration_start`. The
        cumulative iteration and stage
        histories (``_iter_count`` / ``_iter_total_seconds`` and
        ``_stage_count`` / ``_stage_total_seconds``) are deliberately
        preserved so the shared :class:`_HistoryEtaColumn` keeps
        producing a real ETA across trial boundaries. Only the
        transient start timestamps are cleared, covering the case
        where a prior iteration or stage was aborted before its
        matching end hook fired.

        The stage task is added **after** the iteration task on the
        first benchmark so Rich renders trial -> iteration -> stage
        top-to-bottom. On subsequent trials the ``reset`` branch
        preserves the original insertion index, so ordering is not
        re-established per trial; it relies on the protocol invariant
        that ``on_benchmark_start`` fires after ``on_trial_start`` (see
        :class:`ProgressObserver` docstring).
        """
        self._iter_start = None
        self._stage_start = None
        if self._iter_task_id is None:
            self._iter_task_id = self._progress.add_task(
                "Iteration",
                total=total_iterations,
                completed=0,
            )
        else:
            self._progress.reset(
                self._iter_task_id,
                total=total_iterations,
                completed=0,
                description="Iteration",
            )
        if self._stage_task_id is None:
            self._stage_task_id = self._progress.add_task(
                "Stage",
                total=self._stages_per_iteration,
                completed=0,
            )
        else:
            self._progress.reset(
                self._stage_task_id,
                total=self._stages_per_iteration,
                completed=0,
                description="Stage",
            )
        self._refresh()

    def on_iteration_start(
        self,
        iteration: int,  # noqa: ARG002 start timestamp is the signal
    ) -> None:
        """Record the iteration start timestamp and re-zero the stage bar.

        The stage bar is reset back to
        ``0/self._stages_per_iteration`` here (in addition to the
        trial-boundary reset in :meth:`on_benchmark_start`) so its
        ``TimeElapsedColumn`` shows time-into-this-iteration -- the
        operator-useful scoping for a per-iteration bar. A prior
        iteration may have aborted mid-stage, so ``_stage_start`` is
        also cleared defensively.
        """
        self._iter_start = time.monotonic()
        self._stage_start = None
        if self._stage_task_id is not None:
            self._progress.reset(
                self._stage_task_id,
                total=self._stages_per_iteration,
                completed=0,
                description="Stage",
            )
        self._refresh()

    def on_iteration_end(
        self,
        iteration: int,  # noqa: ARG002 advance is the signal
    ) -> None:
        """Advance the iteration bar by one completed iteration.

        ``_stage_start`` is cleared defensively in case the iteration
        aborted mid-stage: a half-run stage is **not** accumulated into
        ``_stage_total_seconds`` -- dropping aborted samples keeps the
        stage ETA honest. The stage bar itself is not reset here;
        :meth:`on_iteration_start` (or :meth:`on_benchmark_start` for
        the next trial) owns that, and leaving the bar full on the
        final iteration of a successful trial is informative.
        """
        if self._iter_start is not None:
            self._iter_total_seconds += time.monotonic() - self._iter_start
            self._iter_count += 1
            self._iter_start = None
        self._stage_start = None
        if self._iter_task_id is not None:
            self._progress.update(self._iter_task_id, advance=1)
        self._refresh()

    def on_stage_start(
        self,
        stage: str,
        iteration: int,  # noqa: ARG002 stage carries the signal
    ) -> None:
        """Paint the stage label on the stage bar and seed its ETA timer."""
        self._stage_start = time.monotonic()
        if self._stage_task_id is not None:
            self._progress.update(
                self._stage_task_id,
                description=f"Stage [{stage}]",
            )
        self._refresh()

    def on_stage_end(
        self,
        stage: str,  # noqa: ARG002 description rolls back on next start
        iteration: int,  # noqa: ARG002
    ) -> None:
        """Accumulate the stage sample into ETA history and advance the stage bar.

        The advance is guarded so a stray extra ``on_stage_end`` in
        the same iteration (a future hook-pairing bug) cannot
        overshoot ``self._stages_per_iteration``; Rich does not clamp
        ``completed`` on ``advance``.
        """
        if self._stage_start is not None:
            self._stage_total_seconds += time.monotonic() - self._stage_start
            self._stage_count += 1
            self._stage_start = None
        if self._stage_task_id is not None:
            task = self._task_by_id(self._stage_task_id)
            if task is not None and task.completed < self._stages_per_iteration:
                self._progress.update(self._stage_task_id, advance=1)
        self._refresh()


def make_observer(
    *,
    enabled: bool,
    console: Console,
    objectives: ObjectivesSection | None = None,
    stages: frozenset[StageName] | None = None,
) -> ProgressObserver:
    """Pick the right observer for the current run.

    Args:
        enabled: Master switch. Usually
            ``console.is_terminal and not --no-progress``.
        console: Shared console that any :class:`RichProgressObserver`
            must render through (also the target of ``RichHandler``).
        objectives: Pareto objectives and recommendation weights for
            the active experiment; forwarded to
            :class:`RichProgressObserver` so the live ``Best so far``
            panel ranks by weighted score. ``None`` (the default)
            keeps the throughput-descending fallback.
        stages: Enabled benchmark sub-stages. Sizes the Stage bar so
            disabled stages do not leave it stranded short of
            ``100%`` and hides ``Best so far`` columns whose metrics
            are produced only by disabled stages. ``None`` (the
            default) falls back to
            :data:`~kube_autotuner.models.ALL_STAGES`.

    Returns:
        A :class:`RichProgressObserver` when ``enabled`` is true,
        otherwise a :class:`NullObserver`.
    """
    if enabled:
        return RichProgressObserver(
            console,
            objectives=objectives,
            stages=stages,
        )
    return NullObserver()
