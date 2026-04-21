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
"""

from __future__ import annotations


class JobAttemptError(RuntimeError):
    """A single client Job attempt did not yield a usable result."""


class ResultValidationError(ValueError):
    """A parsed benchmark payload failed sanity checks."""
