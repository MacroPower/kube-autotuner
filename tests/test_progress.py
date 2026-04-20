"""Tests for :mod:`kube_autotuner.progress`."""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING, cast

from rich.console import Console

from kube_autotuner.progress import (
    NullObserver,
    RichProgressObserver,
    make_observer,
)

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
