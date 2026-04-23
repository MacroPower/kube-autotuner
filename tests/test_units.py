"""Unit tests for :mod:`kube_autotuner.units`.

Coverage mirrors the k8s quantity grammar (``k8s.io/apimachinery`` at
``quantity.go:40-59`` and ``suffix.go:113-132``) grouped by category so
the 1:1 alignment with the upstream suffix set is easy to audit.
"""

from __future__ import annotations

import math

import pytest

from kube_autotuner.units import (
    format_coefficient,
    format_duration,
    parse_quantity,
    pick_duration_unit,
    pick_duration_unit_for_series,
)


class TestBinaryIEC:
    def test_ki(self) -> None:
        assert parse_quantity("1Ki") == 1024

    def test_mi(self) -> None:
        assert parse_quantity("1Mi") == 1024**2

    def test_gi(self) -> None:
        assert parse_quantity("1Gi") == 1024**3

    def test_ti(self) -> None:
        assert parse_quantity("1Ti") == 1024**4

    def test_pi(self) -> None:
        assert parse_quantity("1Pi") == 1024**5

    def test_ei(self) -> None:
        assert parse_quantity("1Ei") == 1024**6

    def test_fractional_gi_is_exact(self) -> None:
        assert parse_quantity("1.5Gi") == pytest.approx(1.5 * 1024**3)

    def test_fractional_ei_is_nearest_float(self) -> None:
        """Pin the lossy-precision behaviour at the top of the IEC range.

        ``1.3Ei`` exceeds float64 mantissa precision; we return the
        nearest representable float rather than raising.
        """
        resolved = parse_quantity("1.3Ei")
        expected = 1.3 * 1024**6
        assert resolved == pytest.approx(expected)


class TestDecimalSISubUnit:
    def test_nano(self) -> None:
        assert parse_quantity("1n") == pytest.approx(1e-9)

    def test_micro(self) -> None:
        assert parse_quantity("1u") == pytest.approx(1e-6)

    def test_milli(self) -> None:
        assert parse_quantity("1m") == pytest.approx(1e-3)

    def test_milli_coefficient(self) -> None:
        assert parse_quantity("500m") == pytest.approx(0.5)


class TestDecimalSISuperUnit:
    def test_k(self) -> None:
        assert parse_quantity("1k") == 1000

    def test_m_mega(self) -> None:
        assert parse_quantity("1M") == pytest.approx(1e6)

    def test_g(self) -> None:
        assert parse_quantity("1G") == pytest.approx(1e9)

    def test_t(self) -> None:
        assert parse_quantity("1T") == pytest.approx(1e12)

    def test_p(self) -> None:
        assert parse_quantity("1P") == pytest.approx(1e15)

    def test_e(self) -> None:
        assert parse_quantity("1E") == pytest.approx(1e18)


class TestDecimalExponentSuffix:
    """k8s treats ``e<N>`` / ``E<N>`` as a suffix, not number-part sci-notation."""

    def test_lowercase_e(self) -> None:
        assert parse_quantity("1e3") == pytest.approx(1000.0)

    def test_uppercase_e(self) -> None:
        assert parse_quantity("1E6") == pytest.approx(1e6)

    def test_negative_exponent(self) -> None:
        assert parse_quantity("1e-6") == pytest.approx(1e-6)

    def test_fractional_base(self) -> None:
        assert parse_quantity("2.5e3") == pytest.approx(2500.0)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1e-06", 1e-6),
            ("1e-09", 1e-9),
        ],
    )
    def test_leading_zero_exponent(self, text: str, expected: float) -> None:
        """Pin round-trip of the normalized-default shapes.

        ``str(float)`` emits a leading zero in the exponent for
        sub-unit defaults (e.g. ``str(1e-6) == "1e-06"``); the parser
        must accept that shape so re-validation of normalized defaults
        is a fixed point.
        """
        assert parse_quantity(text) == pytest.approx(expected)


class TestNumberShapes:
    def test_leading_dot(self) -> None:
        assert parse_quantity(".5") == pytest.approx(0.5)

    def test_trailing_dot(self) -> None:
        assert parse_quantity("1.") == pytest.approx(1.0)

    def test_embedded_dot(self) -> None:
        assert parse_quantity("1.5") == pytest.approx(1.5)


class TestBareAndSigned:
    def test_integer(self) -> None:
        assert parse_quantity("42") == pytest.approx(42.0)

    def test_zero(self) -> None:
        assert parse_quantity("0") == pytest.approx(0.0)

    def test_negative_with_suffix(self) -> None:
        assert parse_quantity("-1Gi") == -(1024**3)

    def test_positive_with_suffix(self) -> None:
        assert parse_quantity("+1Gi") == 1024**3


class TestCaseSensitivity:
    def test_milli_and_mega_distinct(self) -> None:
        assert parse_quantity("1m") != parse_quantity("1M")
        assert parse_quantity("1m") == pytest.approx(1e-3)
        assert parse_quantity("1M") == pytest.approx(1e6)

    @pytest.mark.parametrize("bad", ["1K", "1ki", "1mi", "1gi"])
    def test_rejected_cases(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Cannot parse quantity"):
            parse_quantity(bad)


class TestMixedFormRejection:
    """k8s's two-part split regex rejects these by construction."""

    @pytest.mark.parametrize("bad", ["1e3Ki", "1Mi5", "1KiMi"])
    def test_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Cannot parse quantity"):
            parse_quantity(bad)


class TestWhitespace:
    def test_leading_and_trailing_whitespace_allowed(self) -> None:
        """Pin the deliberate divergence from k8s on whitespace.

        YAML strips whitespace from scalars but the existing constraint
        regex tolerated it, so we preserve that behaviour.
        """
        assert parse_quantity("  64Mi  ") == 64 * 1024**2

    def test_internal_whitespace_rejected(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse quantity"):
            parse_quantity("1 Gi")


class TestMalformed:
    @pytest.mark.parametrize("bad", ["abc", "", "   ", "Gi", "1Xi"])
    def test_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Cannot parse quantity"):
            parse_quantity(bad)


class TestFormatCoefficient:
    @pytest.mark.parametrize(
        ("coeff", "expected"),
        [
            (0.0, "0"),
            (5.0, "5"),
            (5.23, "5.23"),
            (1200.0, "1200"),
            (1.5, "1.5"),
            (0.1, "0.1"),
            (999.5, "999.5"),
        ],
    )
    def test_fixed_point_strips_trailing_zeros(
        self,
        coeff: float,
        expected: str,
    ) -> None:
        """Values render as fixed-point with trailing zeros and dots stripped."""
        assert format_coefficient(coeff) == expected

    def test_no_scientific_notation_at_seconds_scale(self) -> None:
        """Pin the behaviour that motivated the helper's existence.

        ``f"{1200:.3g}"`` produces ``"1.2e+03"`` which is user-hostile
        when suffixed with the seconds unit. The helper must emit a
        plain fixed-point integer instead.
        """
        assert "e" not in format_coefficient(1200.0)
        assert "e" not in format_coefficient(9999.0)


class TestPickDurationUnit:
    @pytest.mark.parametrize(
        ("seconds", "scale", "suffix"),
        [
            (0.0, 1.0, "s"),
            (1.0, 1.0, "s"),
            (60.0, 1.0, "s"),
            (1200.0, 1.0, "s"),
            (0.5, 1e-3, "ms"),
            (1e-3, 1e-3, "ms"),
            (0.005, 1e-3, "ms"),
            (1e-6, 1e-6, "us"),
            (5e-6, 1e-6, "us"),
            (1e-9, 1e-9, "ns"),
            (5e-9, 1e-9, "ns"),
            (5e-10, 1e-9, "ns"),
        ],
    )
    def test_picks_largest_prefix_where_coefficient_at_least_one(
        self,
        seconds: float,
        scale: float,
        suffix: str,
    ) -> None:
        """The picker returns the largest prefix whose scaled coefficient is >= 1."""
        picked_scale, picked_suffix = pick_duration_unit(seconds)
        assert picked_scale == scale
        assert picked_suffix == suffix

    def test_sub_nanosecond_clamps_to_ns(self) -> None:
        """Values below 1ns still render with a unit (do not fall off the scale)."""
        scale, suffix = pick_duration_unit(1e-12)
        assert suffix == "ns"
        assert scale == pytest.approx(1e-9)

    def test_negative_uses_absolute_magnitude(self) -> None:
        """Best-effort: negative inputs pick the unit of their magnitude."""
        assert pick_duration_unit(-0.005) == (1e-3, "ms")


class TestPickDurationUnitForSeries:
    def test_max_magnitude_drives_selection(self) -> None:
        """The picked unit is driven by the series' largest absolute value."""
        scale, suffix = pick_duration_unit_for_series(
            [0.000_001, 0.005, 0.0001],
        )
        assert (scale, suffix) == (1e-3, "ms")

    def test_all_nan_defaults_to_ms(self) -> None:
        """Degenerate case: every value NaN - column still carries ms."""
        scale, suffix = pick_duration_unit_for_series(
            [math.nan, math.nan, math.nan],
        )
        assert (scale, suffix) == (1e-3, "ms")

    def test_mixed_nan_uses_finite_max(self) -> None:
        """NaN entries are skipped; only finite values drive the pick."""
        scale, suffix = pick_duration_unit_for_series(
            [math.nan, 0.000_005, math.nan],
        )
        assert (scale, suffix) == (1e-6, "us")

    def test_all_zero_picks_seconds(self) -> None:
        """An all-zero series yields the seconds bucket via pick_duration_unit."""
        scale, suffix = pick_duration_unit_for_series([0.0, 0.0])
        assert (scale, suffix) == (1.0, "s")

    def test_inf_entries_skipped(self) -> None:
        """Infinity is treated like NaN and skipped."""
        scale, suffix = pick_duration_unit_for_series(
            [math.inf, 0.000_128, -math.inf],
        )
        assert (scale, suffix) == (1e-6, "us")


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "0s"),
            (0.005, "5ms"),
            (0.00523, "5.23ms"),
            (5e-6, "5us"),
            (5e-9, "5ns"),
            (1.5, "1.5s"),
            (5.0, "5s"),
            (1200.0, "1200s"),
            (1.0, "1s"),
            (1e-3, "1ms"),
        ],
    )
    def test_rendering(self, seconds: float, expected: str) -> None:
        """End-to-end rendering matches the documented examples."""
        assert format_duration(seconds) == expected

    def test_user_request_example(self) -> None:
        """Pin the specific example from the user request: 0.005s -> "5ms"."""
        assert format_duration(0.005) == "5ms"

    def test_no_scientific_notation_at_seconds_scale(self) -> None:
        """Regression: large seconds values must not drop into scientific notation."""
        assert "e" not in format_duration(1200.0)
        assert format_duration(1200.0) == "1200s"
