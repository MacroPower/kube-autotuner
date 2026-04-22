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
``kube_autotuner.analysis`` / ``kube_autotuner.plots`` /
``kube_autotuner.report``. Keep the import set to ``rich``, the
standard library, and pure-typing helpers.
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
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from kube_autotuner.scoring import METRIC_TO_DF_COLUMN, score_rows

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
    from kube_autotuner.models import TrialResult


_TOP_N = 5


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
        a previous session's JSONL. The default implementation is a
        no-op; :class:`RichProgressObserver` overrides it to rebuild
        the "best so far" table so the live display does not start
        blank after a resume.

        Args:
            prior: Successful :class:`TrialResult` records from the
                prior session, in file order.
            n_sobol: The Sobol initialization budget for the
                experiment; used to infer each seeded trial's phase
                label from its index.
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
        phase: str,
        metrics: Mapping[str, tuple[float, float]],
    ) -> None:
        """Report that trial ``index`` finished successfully.

        Args:
            index: Zero-based trial index.
            phase: ``"sobol"`` or ``"bayesian"``.
            metrics: The per-metric ``(mean, SEM)`` table returned by
                the optimizer's metric aggregation.
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
        phase: str,
        metrics: Mapping[str, tuple[float, float]],
    ) -> None:
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
    2. *Trials bar.* ``Task.speed`` requires **at least two** progress
       samples to produce a value (independent of pruning). With only
       one ``advance=1`` call posted after the first completion, the
       column would render ``-:--:--`` until the second trial finished
       -- roughly 19 minutes into a 9.5-min-per-trial run.

    Both bars supply a ``(count, total_seconds)`` snapshot through a
    caller-provided callable, letting the ETA be available after the
    very first completion and span the entire run.
    """

    def __init__(self, history: Callable[[], tuple[int, float, float]]) -> None:
        """Bind the column to a ``(count, total_seconds, in_flight)`` supplier.

        Args:
            history: Zero-arg callable returning the observer's
                cumulative completed count, total observed duration in
                seconds, and the elapsed time of the currently
                in-flight unit (``0.0`` when no unit is active). The
                in-flight component lets Rich's 8 Hz refresh tick the
                ETA down second-by-second within a unit instead of
                holding flat until the next completion.
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
        count, total_seconds, in_flight = self._history()
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
        "cpu",
        "index",
        "jitter_ms",
        "metrics",
        "node_memory_mib",
        "p99_ms",
        "phase",
        "retx_per_mb",
        "rps",
        "throughput_mbps",
    )

    def __init__(  # noqa: PLR0913 keyword-only summary bundle for one trial row
        self,
        *,
        index: int,
        phase: str,
        throughput_mbps: float,
        cpu: float,
        retx_per_mb: float,
        jitter_ms: float,
        rps: float,
        node_memory_mib: float,
        p99_ms: float,
        metrics: dict[str, float],
    ) -> None:
        """Store one trial's summary for display and scoring.

        Args:
            index: Zero-based trial index within this run.
            phase: ``"sobol"`` or ``"bayesian"``.
            throughput_mbps: Mean throughput in megabits-per-second.
            cpu: Mean target-node CPU utilization, 0-100.
            retx_per_mb: Retransmits per megabyte; ``NaN`` when the
                trial produced no TCP bytes (every ``bw-tcp`` stage
                failed).
            jitter_ms: Mean UDP jitter in milliseconds from the
                ``bw-udp`` stage; ``NaN`` when no UDP record
                reported jitter.
            rps: Mean requests-per-second from the fortio saturation
                sub-stage; ``NaN`` when the trial produced no
                saturation records.
            node_memory_mib: Mean target-node memory usage in MiB;
                ``NaN`` when the trial produced no memory samples.
            p99_ms: Fixed-QPS p99 latency in milliseconds; ``NaN``
                when the trial produced no fortio records.
            metrics: Raw-domain per-metric means keyed by the
                :data:`kube_autotuner.scoring.METRIC_TO_DF_COLUMN`
                DataFrame column name (``mean_throughput``,
                ``mean_cpu``, ...). Fed to
                :func:`kube_autotuner.scoring.score_rows` for the
                live top-N ranking, which min-max normalizes across
                every stored row on each refresh. Stores ``NaN`` for
                metrics the trial did not observe.
        """
        self.index = index
        self.phase = phase
        self.throughput_mbps = throughput_mbps
        self.cpu = cpu
        self.retx_per_mb = retx_per_mb
        self.jitter_ms = jitter_ms
        self.rps = rps
        self.node_memory_mib = node_memory_mib
        self.p99_ms = p99_ms
        self.metrics = metrics


def _build_trial_row(  # noqa: PLR0914 one-pass projection over the full metric bundle
    index: int,
    phase: str,
    metrics: Mapping[str, tuple[float, float]],
) -> _TrialRow:
    """Project a completed-trial metric bundle into a :class:`_TrialRow`.

    Handles two concerns in one pass:

    * derives the display-domain floats (Mbps / MiB / retx/MB) for
      the rendered table, and
    * captures the full raw-domain metric bundle keyed by
      :data:`kube_autotuner.scoring.METRIC_TO_DF_COLUMN` so
      :func:`kube_autotuner.scoring.score_rows` can min-max normalize
      across every metric the user's objectives reference.

    Without the raw bundle, metrics not surfaced by the display
    (``cni_memory``, ``rps``, ``latency_p50`` / ``p90`` / ``p99``)
    would be NaN across every row and collapse to the
    degenerate-column fallback, making the live ranking diverge
    from :func:`kube_autotuner.analysis.recommend_configs`.

    Args:
        index: Zero-based trial index within this run.
        phase: ``"sobol"`` or ``"bayesian"``.
        metrics: The ``(mean, SEM)`` bundle from the optimizer.

    Returns:
        A :class:`_TrialRow` ready to append to ``_all_rows``.
    """
    tp_pair = metrics.get("throughput")
    cpu_pair = metrics.get("cpu")
    rate_pair = metrics.get("retransmit_rate")
    jitter_pair = metrics.get("jitter")
    rps_pair = metrics.get("rps")
    nmem_pair = metrics.get("node_memory")
    p99_pair = metrics.get("latency_p99")
    tp = (tp_pair[0] if tp_pair is not None else 0.0) / 1e6
    cpu = cpu_pair[0] if cpu_pair is not None else 0.0
    rate = rate_pair[0] if rate_pair is not None else math.nan
    retx_per_mb = math.nan if math.isnan(rate) else rate * 1e6
    jitter = jitter_pair[0] if jitter_pair is not None else math.nan
    rps = rps_pair[0] if rps_pair is not None else math.nan
    nmem_bytes = nmem_pair[0] if nmem_pair is not None else math.nan
    node_memory_mib = math.nan if math.isnan(nmem_bytes) else nmem_bytes / (1024 * 1024)
    p99 = p99_pair[0] if p99_pair is not None else math.nan
    raw_metrics: dict[str, float] = {
        METRIC_TO_DF_COLUMN[key]: (pair[0] if pair is not None else math.nan)
        for key, pair in (
            ("throughput", tp_pair),
            ("cpu", cpu_pair),
            ("retransmit_rate", rate_pair),
            ("jitter", jitter_pair),
            ("node_memory", nmem_pair),
            ("cni_memory", metrics.get("cni_memory")),
            ("rps", rps_pair),
            ("latency_p50", metrics.get("latency_p50")),
            ("latency_p90", metrics.get("latency_p90")),
            ("latency_p99", p99_pair),
        )
    }
    return _TrialRow(
        index=index,
        phase=phase,
        throughput_mbps=tp,
        cpu=cpu,
        retx_per_mb=retx_per_mb,
        jitter_ms=jitter,
        rps=rps,
        node_memory_mib=node_memory_mib,
        p99_ms=p99,
        metrics=raw_metrics,
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
    """Live ``rich`` display with a trials bar, iteration bar, and results table.

    The observer must be entered as a context manager. It is safe to
    re-enter sequentially (construct, enter, exit, enter again) but
    not nested.
    """

    def __init__(
        self,
        console: Console,
        objectives: ObjectivesSection | None = None,
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
        """
        self._console = console
        self._objectives = objectives
        self._iter_start: float | None = None
        self._iter_count: int = 0
        self._iter_total_seconds: float = 0.0
        self._trial_start: float | None = None
        self._trial_count: int = 0
        self._trial_total_seconds: float = 0.0
        self._trials: Progress = self._make_progress(
            eta_column=_HistoryEtaColumn(self._trial_history),
        )
        self._iters: Progress = self._make_progress(
            eta_column=_HistoryEtaColumn(self._iter_history),
        )
        self._trial_task_id: TaskID | None = None
        self._iter_task_id: TaskID | None = None
        self._all_rows: list[_TrialRow] = []
        self._top: list[_TrialRow] = []
        self._phase: str = "sobol"
        self._current_stage: str = ""
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

    @staticmethod
    def _make_progress(eta_column: ProgressColumn | None = None) -> Progress:
        """Build a :class:`rich.progress.Progress` with the standard columns.

        Args:
            eta_column: Optional replacement for the default
                :class:`TimeRemainingColumn`. Both the trials bar and
                iteration bar supply a :class:`_HistoryEtaColumn` so
                the ETA is available after the first completion and
                survives per-mode resets.

        Returns:
            A fresh :class:`Progress` composed of the bar, M-of-N, and
            elapsed/ETA columns used throughout the live display.
        """
        eta: ProgressColumn = (
            eta_column if eta_column is not None else TimeRemainingColumn()
        )
        return Progress(
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[dim]elapsed[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/dim]"),
            eta,
            expand=True,
        )

    def _render(self) -> Group:
        """Compose the live renderable (two bars stacked over a table).

        The "best so far" table is omitted until at least one trial
        has completed, so the ``baseline`` and ``trial`` runs (which
        never call :meth:`on_trial_complete`) show bars only rather
        than an empty table frame.

        Returns:
            A :class:`rich.console.Group` of the trials bar, the
            iteration bar, and (when any trial has completed) the
            current "best so far" table.
        """
        if not self._top:
            return Group(self._trials, self._iters)
        sort_key = "weighted score" if self._objectives is not None else "throughput"
        table = Table(
            title=f"Best so far (top {_TOP_N} by {sort_key})",
            title_style="bold",
            show_header=True,
            header_style="bold",
            expand=True,
        )
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("Phase", no_wrap=True)
        table.add_column("Throughput", justify="right")
        table.add_column("CPU", justify="right")
        table.add_column("retx/MB", justify="right")
        table.add_column("jitter ms", justify="right")
        table.add_column("RPS", justify="right")
        table.add_column("node mem", justify="right")
        table.add_column("p99 ms", justify="right")
        for row in self._top:
            retx = "n/a" if math.isnan(row.retx_per_mb) else f"{row.retx_per_mb:.2f}"
            jitter = "n/a" if math.isnan(row.jitter_ms) else f"{row.jitter_ms:.3f}"
            rps = "n/a" if math.isnan(row.rps) else f"{row.rps:,.1f}"
            nmem = (
                "n/a"
                if math.isnan(row.node_memory_mib)
                else f"{row.node_memory_mib:,.0f} MiB"
            )
            p99 = "n/a" if math.isnan(row.p99_ms) else f"{row.p99_ms:.1f}"
            table.add_row(
                str(row.index + 1),
                row.phase,
                f"{row.throughput_mbps:,.1f} Mbps",
                f"{row.cpu:.1f}%",
                retx,
                jitter,
                rps,
                nmem,
                p99,
            )
        return Group(self._trials, self._iters, table)

    def _refresh(self) -> None:
        """Stage the current renderable for the next auto-refresh tick."""
        if self._live is not None:
            self._live.update(self._render())

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
        """Create the trials bar on first call, advance the phase label."""
        self._phase = phase
        description = f"Trials [{phase}]"
        if self._trial_task_id is None:
            self._trial_task_id = self._trials.add_task(
                description,
                total=total,
                completed=index,
            )
        else:
            self._trials.update(
                self._trial_task_id,
                description=description,
                total=total,
            )
        self._trial_start = time.monotonic()
        self._refresh()

    def on_trial_complete(
        self,
        index: int,
        phase: str,
        metrics: Mapping[str, tuple[float, float]],
    ) -> None:
        """Advance the trials bar and fold the result into the top-N table."""
        if self._trial_start is not None:
            self._trial_total_seconds += time.monotonic() - self._trial_start
            self._trial_count += 1
            self._trial_start = None
        if self._trial_task_id is not None:
            self._trials.update(self._trial_task_id, advance=1)
        row = _build_trial_row(index, phase, metrics)
        self._all_rows.append(row)
        self._rerank()
        self._refresh()

    def seed_history(self, prior: list[TrialResult], n_sobol: int) -> None:
        """Pre-populate the "best so far" table from prior trials.

        Called once at loop-start when resuming from a prior session.
        The phase of each seeded row is inferred from its index versus
        ``n_sobol`` -- this is sound because the compat check rejects
        resumes whose ``n_sobol`` differs from the sidecar's.
        ``_trial_count`` / ``_trial_total_seconds`` are deliberately
        left at zero so the ETA re-learns from live wall-clock; the
        first ``on_trial_start`` advances the Rich bar to
        ``completed=index`` to reflect the seeded position.

        Args:
            prior: Prior successful :class:`TrialResult` records in
                file order.
            n_sobol: The current experiment's Sobol budget; used to
                infer the phase label per seeded row.
        """
        # Lazy import: ``_compute_metrics`` lives in ``optimizer`` which
        # is reachable only when the ``optimize`` dep group is
        # installed. Keeping this import inside the method preserves
        # the module-level constraint that ``progress`` stays
        # optimize-free for ``task completions``.
        from kube_autotuner.optimizer import _compute_metrics  # noqa: PLC0415

        for idx, tr in enumerate(prior):
            phase = "sobol" if idx < n_sobol else "bayesian"
            metrics = _compute_metrics(tr)
            self._all_rows.append(_build_trial_row(idx, phase, metrics))
        self._rerank()
        if prior:
            self._phase = "sobol" if len(prior) < n_sobol else "bayesian"

    def _rerank(self) -> None:
        """Refresh ``self._top`` from ``self._all_rows``.

        When an :class:`~kube_autotuner.experiment.ObjectivesSection`
        is bound the ranking goes through
        :func:`kube_autotuner.scoring.score_rows` so the panel
        matches ``recommend_configs``. Without one (bare observer
        construction, ``baseline`` / ``trial`` flows) the legacy
        throughput-descending sort is used.
        """
        if self._objectives is not None:
            scores = score_rows(
                [r.metrics for r in self._all_rows],
                self._objectives.pareto,
                self._objectives.recommendation_weights,
            )
            # Rank by score desc, break ties by arrival order (lower
            # _all_rows index first) to match the stable-mergesort
            # tiebreak in recommend_configs.
            order = sorted(
                range(len(self._all_rows)),
                key=lambda i: (-scores[i], i),
            )
            self._top = [self._all_rows[i] for i in order[:_TOP_N]]
        else:
            self._top = sorted(
                self._all_rows,
                key=lambda r: r.throughput_mbps,
                reverse=True,
            )[:_TOP_N]

    def on_trial_failed(
        self,
        index: int,  # noqa: ARG002 recorded for future failure panel
        exc: BaseException,  # noqa: ARG002
    ) -> None:
        """Advance the trials bar without adding a row to the results table."""
        if self._trial_start is not None:
            self._trial_total_seconds += time.monotonic() - self._trial_start
            self._trial_count += 1
            self._trial_start = None
        if self._trial_task_id is not None:
            self._trials.update(self._trial_task_id, advance=1)
        self._refresh()

    def on_benchmark_start(self, total_iterations: int) -> None:
        """Size the iteration bar for the trial's full iteration budget.

        Fired once per :meth:`BenchmarkRunner.run` call. The iteration
        bar spans every iteration in the trial, so its ``total`` is
        set to ``iterations`` and its ``completed`` starts at zero.
        The cumulative iteration history (``_iter_count`` /
        ``_iter_total_seconds``) is deliberately preserved so the
        custom :class:`_IterEtaColumn` can keep producing a real ETA
        across trial boundaries. Only the transient
        start-of-iteration timestamp is cleared, covering the case
        where a prior iteration was aborted before
        :meth:`on_iteration_end` fired.
        """
        self._iter_start = None
        if self._iter_task_id is None:
            self._iter_task_id = self._iters.add_task(
                "Current",
                total=total_iterations,
                completed=0,
            )
        else:
            self._iters.reset(
                self._iter_task_id,
                total=total_iterations,
                completed=0,
                description="Current",
            )
        self._refresh()

    def on_iteration_start(
        self,
        iteration: int,  # noqa: ARG002 start timestamp is the signal
    ) -> None:
        """Record the iteration start timestamp for ETA tracking.

        The iteration-bar description is repainted by the immediately
        following :meth:`on_stage_start`, so no description update
        happens here.
        """
        self._iter_start = time.monotonic()

    def on_iteration_end(
        self,
        iteration: int,  # noqa: ARG002 advance is the signal
    ) -> None:
        """Advance the iteration bar by one completed iteration."""
        if self._iter_start is not None:
            self._iter_total_seconds += time.monotonic() - self._iter_start
            self._iter_count += 1
            self._iter_start = None
        if self._iter_task_id is not None:
            self._iters.update(self._iter_task_id, advance=1)
        self._current_stage = ""
        self._refresh()

    def on_stage_start(
        self,
        stage: str,
        iteration: int,  # noqa: ARG002 stage carries the signal
    ) -> None:
        """Update the iteration-bar description to reflect the active sub-stage."""
        self._current_stage = stage
        if self._iter_task_id is not None:
            self._iters.update(
                self._iter_task_id,
                description=f"Current [{stage}]",
            )
        self._refresh()

    def on_stage_end(
        self,
        stage: str,  # noqa: ARG002 description rolls back on next start
        iteration: int,  # noqa: ARG002
    ) -> None:
        """Clear the sub-stage label; kept as a hook for future extensions."""
        self._current_stage = ""
        self._refresh()


def make_observer(
    *,
    enabled: bool,
    console: Console,
    objectives: ObjectivesSection | None = None,
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

    Returns:
        A :class:`RichProgressObserver` when ``enabled`` is true,
        otherwise a :class:`NullObserver`.
    """
    if enabled:
        return RichProgressObserver(console, objectives=objectives)
    return NullObserver()
