"""k8s-style quantity parsing for outcome-constraint thresholds.

Grammar mirrors ``k8s.io/apimachinery/pkg/api/resource/quantity.go`` (BNF
at ``quantity.go:40-59``) 1:1 for the accepted input set. Returned values
are ``float`` rather than k8s's int64+infDec; integer-valued quantities
are exact up to ``2**53``, and fractional IEC coefficients are exact up
to and including ``Ti``. Intentional divergences: no canonical
re-emission (callers stringify for Ax, which accepts bare floats only),
and permissive leading/trailing whitespace.

See also :func:`kube_autotuner.benchmark.parser.parse_k8s_memory` for the
int-bytes variant used by the benchmark runner for pod memory strings.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_SUFFIXES: dict[str, int | float] = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,
    "Ei": 1024**6,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1_000,
    "M": 1_000_000,
    "G": 1_000_000_000,
    "T": 1_000_000_000_000,
    "P": 1_000_000_000_000_000,
    "E": 1_000_000_000_000_000_000,
}

NUMBER_PATTERN: str = r"[+-]?(?:\d+\.?\d*|\.\d+)"
SUFFIX_PATTERN: str = r"(?:[eE][+-]?\d+)|Ki|Mi|Gi|Ti|Pi|Ei|[numkMGTPE]"

_QUANTITY_RE = re.compile(
    rf"^\s*(?P<value>{NUMBER_PATTERN})(?P<suffix>{SUFFIX_PATTERN})?\s*$",
)


def parse_quantity(s: str) -> float:
    """Parse a k8s-style quantity to a numeric value.

    Accepts the full k8s quantity grammar: bare numbers (``"42"``,
    ``".5"``, ``"1."``), binary IEC suffixes (``Ki|Mi|Gi|Ti|Pi|Ei``),
    decimal SI suffixes (``n|u|m|k|M|G|T|P|E``), and decimal-exponent
    suffixes (``e<N>`` / ``E<N>``, e.g. ``"1e3"`` = 1000, ``"1E-6"`` =
    1e-6). Suffix matching is case-sensitive: ``m`` differs from ``M``,
    and ``K`` / ``ki`` / ``mi`` / ``gi`` are rejected, matching k8s
    semantics. Mixed forms such as ``"1e3Ki"`` are rejected (the
    decimal-exponent suffix consumes ``e3``, leaving ``Ki`` unmatched
    against the anchor).

    Args:
        s: Quantity string. Whitespace around the whole token is
            stripped; the numeric portion and the suffix must be
            contiguous (``"1Gi"`` parses, ``"1 Gi"`` does not).

    Returns:
        The resolved numeric value as ``float``. Integer-valued results
        are still returned as ``float`` for uniform downstream handling.

    Raises:
        ValueError: When ``s`` does not match
            ``<signedNumber><optional-suffix>``.
    """
    match = _QUANTITY_RE.match(s)
    if match is None:
        msg = f"Cannot parse quantity {s!r}"
        raise ValueError(msg)
    value = float(match.group("value"))
    suffix = match.group("suffix")
    if suffix is None:
        return value
    if suffix in _SUFFIXES:
        return value * _SUFFIXES[suffix]
    # Decimal-exponent form (``e<N>`` / ``E<N>``); single-char ``E``
    # (Exa) is handled by the dict branch above.
    return value * 10.0 ** int(suffix[1:])


def seconds_to_ms(v: float) -> float:
    """Convert a seconds value to milliseconds for display.

    Args:
        v: Seconds value; callers guard on ``None`` before calling.

    Returns:
        The input scaled by ``1000.0``.
    """
    return v * 1000.0


_DURATION_UNITS: tuple[tuple[float, str], ...] = (
    (1.0, "s"),
    (1e-3, "ms"),
    (1e-6, "us"),
    (1e-9, "ns"),
)

# Below this magnitude the input is treated as zero. Smaller than the
# smallest unit (1 ns = 1e-9) by many orders of magnitude so picker
# behaviour for any realistic duration is unchanged.
_DURATION_ZERO_THRESHOLD: float = 1e-30


def format_coefficient(coeff: float) -> str:
    """Format a scaled coefficient as fixed-point with trailing zeros stripped.

    Fixed-point rather than ``f"{x:.3g}"`` because ``.3g`` drops into
    scientific notation at ``coeff >= 1000`` - user-hostile for the
    seconds suffix where values can grow unboundedly.

    Args:
        coeff: Scaled coefficient (value divided by the picked
            unit's scale). NaN / infinity are caller responsibility.

    Returns:
        The coefficient rendered with up to three decimal places and
        trailing zeros / trailing decimal point stripped. Returns
        ``"0"`` for zero so the output is never empty.
    """
    s = f"{coeff:.3f}".rstrip("0").rstrip(".")
    return s or "0"


def pick_duration_unit(seconds: float) -> tuple[float, str]:
    """Return ``(scale, suffix)`` for the largest SI prefix with coefficient >= 1.

    Sub-second SI only: values whose absolute magnitude is ``>= 1``
    always pick ``(1.0, "s")`` rather than shifting into ``ks`` /
    ``Ms``. Callers must pre-filter ``NaN`` / ``None``. Negative
    inputs are not expected (durations are non-negative by
    construction); magnitude selection uses ``abs()`` as best-effort.

    Args:
        seconds: A seconds value; ``NaN`` / ``None`` are the caller's
            responsibility to filter before calling.

    Returns:
        A ``(scale, suffix)`` tuple where ``scale`` is the divisor
        applied to ``seconds`` to produce the display coefficient and
        ``suffix`` is one of ``"s"``, ``"ms"``, ``"us"``, ``"ns"``.
        Zero returns ``(1.0, "s")``. Values below ``1ns`` clamp to
        ``(1e-9, "ns")`` so sub-nanosecond inputs still render with
        a unit.
    """
    magnitude = abs(seconds)
    if magnitude < _DURATION_ZERO_THRESHOLD:
        return (1.0, "s")
    for scale, suffix in _DURATION_UNITS:
        if magnitude >= scale:
            return (scale, suffix)
    return _DURATION_UNITS[-1]


def pick_duration_unit_for_series(values: Iterable[float]) -> tuple[float, str]:
    """Return ``(scale, suffix)`` chosen from the max finite magnitude of ``values``.

    Used for tabular displays where every cell in a column must share
    one unit so they stay comparable. The picked unit is driven by
    the column's largest-magnitude row: a column whose max is
    ``0.0015`` renders every cell in ms; a column whose max is
    ``1.5`` renders every cell in seconds.

    Args:
        values: An iterable of seconds values. ``NaN`` and infinity
            entries are skipped.

    Returns:
        A ``(scale, suffix)`` tuple produced by
        :func:`pick_duration_unit` from the max absolute finite
        value. Defaults to ``(1e-3, "ms")`` when every value is
        non-finite so the column still carries a unit.
    """
    max_magnitude = 0.0
    any_finite = False
    for v in values:
        if math.isnan(v) or math.isinf(v):
            continue
        any_finite = True
        m = abs(v)
        max_magnitude = max(max_magnitude, m)
    if not any_finite:
        return (1e-3, "ms")
    return pick_duration_unit(max_magnitude)


def format_duration(seconds: float) -> str:
    """Render a seconds value with the unit picked by :func:`pick_duration_unit`.

    Uses fixed-point formatting with up to three decimal places and
    trailing zeros / trailing decimal point stripped, so
    ``0.005 -> "5ms"``, ``0.00523 -> "5.23ms"``, ``1.5 -> "1.5s"``,
    ``1200 -> "1200s"``.

    Args:
        seconds: A seconds value. ``NaN`` / ``None`` are the caller's
            responsibility to filter before calling.

    Returns:
        The value rendered as ``"<coefficient><suffix>"`` where
        ``<suffix>`` is one of ``"s"``, ``"ms"``, ``"us"``, ``"ns"``.
    """
    scale, suffix = pick_duration_unit(seconds)
    return f"{format_coefficient(seconds / scale)}{suffix}"
