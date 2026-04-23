"""Exceptions raised by the benchmark runner and its parsers.

``JobAttemptError`` marks a *per-attempt* failure inside the runner's
retry loop (e.g. the Job entered ``Failed=True``, no ``Succeeded`` pod
was present, or the payload failed validation). It does **not** mean the
whole benchmark trial has failed: the runner catches it and retries up
to the configured ``max_attempts``.

``ResultValidationError`` marks a parsed payload that is structurally
valid JSON but semantically degenerate (iperf3 reported an ``error``
field, fortio histogram is empty, etc.). It subclasses :class:`ValueError`
to align with :func:`kube_autotuner.benchmark.fortio_parser.extract_fortio_result_json`
which already raises plain ``ValueError`` on a missing result block; the
runner catches both via a single ``except ResultValidationError`` arm
after wrapping the parser's bare ``ValueError`` at the call site.

These two types are deliberately unrelated by inheritance. A
``JobAttemptError`` is an operational failure (the system did a thing
and it did not work); a ``ResultValidationError`` is a value error (the
system did a thing, got a value back, and the value was bad). A shared
base would blur the semantics and tempt callers to catch too broadly.

``ClientJobFailed`` and ``BenchmarkFailure`` are the two diagnostic
envelopes the runner raises at the retry-loop and stage boundary
respectively. They carry structured per-attempt and per-server
snapshots so the optimizer can persist a failure dump without having
to reach into runner internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kube_autotuner.k8s.client import JobFailureDiagnostics


class JobAttemptError(RuntimeError):
    """A single client Job attempt did not yield a usable result."""


class ResultValidationError(ValueError):
    """A parsed benchmark payload failed sanity checks."""


class ClientJobFailed(RuntimeError):  # noqa: N818 - envelope, not a bare error type
    """A client Job exhausted its retry budget.

    Carries the structured per-attempt diagnostics captured by
    :meth:`kube_autotuner.benchmark.runner.BenchmarkRunner._log_job_diagnostics`
    so the stage method can fold them into a :class:`BenchmarkFailure`
    for the optimizer to persist.

    Attributes:
        diagnostics: One :class:`JobFailureDiagnostics` entry per
            failed attempt, in attempt order. May be empty if every
            describe call itself raised.
    """

    def __init__(
        self,
        message: str,
        *,
        diagnostics: list[JobFailureDiagnostics],
    ) -> None:
        """Store the diagnostics list alongside the formatted message.

        Args:
            message: Human-readable failure summary.
            diagnostics: Per-attempt diagnostic rows.
        """
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass
class BenchmarkFailure(RuntimeError):  # noqa: N818 - envelope, not a bare error type
    """Wrap a per-trial benchmark failure with structured diagnostics.

    ``@dataclass`` on a :class:`BaseException` subclass does not call
    ``BaseException.__init__(*args)``, so ``self.args`` is ``()`` and
    the default ``BaseException.__str__`` would return ``""``. The
    explicit :meth:`__str__` override below is load-bearing -- do not
    remove it. Cause chaining via
    ``raise BenchmarkFailure(...) from first_exc`` still works because
    it sets ``__cause__`` independently of ``args``.

    Attributes:
        cause: The underlying exception (also attached as ``__cause__``
            via ``raise ... from``).
        attempt_diagnostics: Per-attempt :class:`JobFailureDiagnostics`
            rows for the failed client Job, in attempt order.
        server_snapshots: One row per server pod sampled at stage
            failure time.
        stage: Sub-stage label, e.g. ``"bw-udp"`` / ``"fortio-sat"``.
        iteration: Zero-based iteration index.
    """

    cause: BaseException
    attempt_diagnostics: list[JobFailureDiagnostics] = field(default_factory=list)
    server_snapshots: list[dict[str, Any]] = field(default_factory=list)
    stage: str = ""
    iteration: int = -1

    def __str__(self) -> str:
        """Render a concise one-line summary; see class docstring caveat.

        Returns:
            ``"benchmark failure in <stage> iter <i>: <cause>"``.
        """
        return f"benchmark failure in {self.stage} iter {self.iteration}: {self.cause}"
