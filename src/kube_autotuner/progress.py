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
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, Self

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    from rich.console import Console
    from rich.progress import TaskID


_TOP_N = 5


class ProgressObserver(Protocol):
    """Progress callbacks emitted by the optimize / benchmark loops.

    Every method is fire-and-forget: implementations must not raise
    and must tolerate out-of-order or missing calls (for example,
    ``on_mode_start`` is not guaranteed before ``on_iteration_start``
    for callers that do not know the mode structure up front).

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

    def on_mode_start(self, mode: str, total_iterations: int) -> None:
        """Report that the benchmark is starting a new iperf3 mode.

        Args:
            mode: ``"tcp"`` or ``"udp"``.
            total_iterations: Iterations configured for this mode.
        """
        ...

    def on_iteration_start(self, mode: str, iteration: int) -> None:
        """Report that a single benchmark iteration is starting.

        Args:
            mode: ``"tcp"`` or ``"udp"``.
            iteration: Zero-based iteration index within ``mode``.
        """
        ...

    def on_iteration_end(self, mode: str, iteration: int) -> None:
        """Report that a single benchmark iteration has finished.

        Args:
            mode: ``"tcp"`` or ``"udp"``.
            iteration: Zero-based iteration index within ``mode``.
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

    def on_mode_start(self, mode: str, total_iterations: int) -> None:
        """No-op."""

    def on_iteration_start(self, mode: str, iteration: int) -> None:
        """No-op."""

    def on_iteration_end(self, mode: str, iteration: int) -> None:
        """No-op."""


class _TrialRow:
    """One row of the live "best so far" table."""

    __slots__ = ("cpu", "index", "phase", "retx_per_mb", "throughput_mbps")

    def __init__(
        self,
        *,
        index: int,
        phase: str,
        throughput_mbps: float,
        cpu: float,
        retx_per_mb: float,
    ) -> None:
        """Store one trial's summary for display.

        Args:
            index: Zero-based trial index within this run.
            phase: ``"sobol"`` or ``"bayesian"``.
            throughput_mbps: Mean throughput in megabits-per-second.
            cpu: Mean target-node CPU utilization, 0-100.
            retx_per_mb: Retransmits per megabyte; ``NaN`` when the
                mode cannot observe it (e.g. pure UDP).
        """
        self.index = index
        self.phase = phase
        self.throughput_mbps = throughput_mbps
        self.cpu = cpu
        self.retx_per_mb = retx_per_mb


class RichProgressObserver:
    """Live ``rich`` display with a trials bar, iteration bar, and results table.

    The observer must be entered as a context manager. It is safe to
    re-enter sequentially (construct, enter, exit, enter again) but
    not nested.
    """

    def __init__(self, console: Console) -> None:
        """Bind the observer to a shared ``rich`` console.

        Args:
            console: Console to render through. Must be the same
                instance passed to any ``RichHandler`` attached to
                the root logger so log records interleave above the
                live region instead of tearing through it.
        """
        self._console = console
        self._trials: Progress = self._make_progress()
        self._iters: Progress = self._make_progress()
        self._trial_task_id: TaskID | None = None
        self._iter_task_id: TaskID | None = None
        self._top: list[_TrialRow] = []
        self._phase: str = "sobol"
        self._current_mode: str = ""
        self._live: Live | None = None

    @staticmethod
    def _make_progress() -> Progress:
        """Build a :class:`rich.progress.Progress` with the standard columns.

        Returns:
            A fresh :class:`Progress` composed of the bar, M-of-N, and
            elapsed/ETA columns used throughout the live display.
        """
        return Progress(
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[dim]elapsed[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/dim]"),
            TimeRemainingColumn(),
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
        table = Table(
            title=f"Best so far (top {_TOP_N} by throughput)",
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
        for row in self._top:
            retx = "n/a" if math.isnan(row.retx_per_mb) else f"{row.retx_per_mb:.2f}"
            table.add_row(
                str(row.index + 1),
                row.phase,
                f"{row.throughput_mbps:,.1f} Mbps",
                f"{row.cpu:.1f}%",
                retx,
            )
        return Group(self._trials, self._iters, table)

    def _refresh(self) -> None:
        """Push the current renderable into the live display."""
        if self._live is not None:
            self._live.update(self._render(), refresh=True)

    def __enter__(self) -> Self:
        """Start the ``rich.live.Live`` display on the bound console.

        Returns:
            The observer instance so callers can use ``with obs as o:``.
        """
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the live display; swallow no exceptions."""
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
            self._live = None

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
        self._refresh()

    def on_trial_complete(
        self,
        index: int,
        phase: str,
        metrics: Mapping[str, tuple[float, float]],
    ) -> None:
        """Advance the trials bar and fold the result into the top-N table."""
        if self._trial_task_id is not None:
            self._trials.update(self._trial_task_id, advance=1)
        tp_pair = metrics.get("throughput")
        cpu_pair = metrics.get("cpu")
        rate_pair = metrics.get("retransmit_rate")
        tp = (tp_pair[0] if tp_pair is not None else 0.0) / 1e6
        cpu = cpu_pair[0] if cpu_pair is not None else 0.0
        rate = rate_pair[0] if rate_pair is not None else math.nan
        retx_per_mb = math.nan if math.isnan(rate) else rate * 1e6
        row = _TrialRow(
            index=index,
            phase=phase,
            throughput_mbps=tp,
            cpu=cpu,
            retx_per_mb=retx_per_mb,
        )
        self._top.append(row)
        self._top.sort(key=lambda r: r.throughput_mbps, reverse=True)
        del self._top[_TOP_N:]
        self._refresh()

    def on_trial_failed(
        self,
        index: int,  # noqa: ARG002 recorded for future failure panel
        exc: BaseException,  # noqa: ARG002
    ) -> None:
        """Advance the trials bar without adding a row to the results table."""
        if self._trial_task_id is not None:
            self._trials.update(self._trial_task_id, advance=1)
        self._refresh()

    def on_mode_start(self, mode: str, total_iterations: int) -> None:
        """Reset the iteration bar for the new mode."""
        self._current_mode = mode
        description = f"Current [{mode}]"
        if self._iter_task_id is None:
            self._iter_task_id = self._iters.add_task(
                description,
                total=total_iterations,
                completed=0,
            )
        else:
            self._iters.reset(
                self._iter_task_id,
                total=total_iterations,
                completed=0,
                description=description,
            )
        self._refresh()

    def on_iteration_start(
        self,
        mode: str,
        iteration: int,  # noqa: ARG002 start is just a label change
    ) -> None:
        """Refresh the iteration-bar description to reflect ``mode``."""
        if self._iter_task_id is not None:
            self._iters.update(
                self._iter_task_id,
                description=f"Current [{mode}]",
            )
        self._refresh()

    def on_iteration_end(
        self,
        mode: str,  # noqa: ARG002 mode already on the task description
        iteration: int,  # noqa: ARG002 advance is the signal
    ) -> None:
        """Advance the iteration bar by one completed iteration."""
        if self._iter_task_id is not None:
            self._iters.update(self._iter_task_id, advance=1)
        self._refresh()


def make_observer(*, enabled: bool, console: Console) -> ProgressObserver:
    """Pick the right observer for the current run.

    Args:
        enabled: Master switch. Usually
            ``console.is_terminal and not --no-progress``.
        console: Shared console that any :class:`RichProgressObserver`
            must render through (also the target of ``RichHandler``).

    Returns:
        A :class:`RichProgressObserver` when ``enabled`` is true,
        otherwise a :class:`NullObserver`.
    """
    if enabled:
        return RichProgressObserver(console)
    return NullObserver()
