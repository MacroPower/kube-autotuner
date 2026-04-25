"""Unit tests for :mod:`kube_autotuner.benchmark.manifests`.

Each wrapper combines a ``build_*_yaml`` builder with
:func:`kube_autotuner.benchmark.patch.apply_patches`. The inner builders
have their own dedicated tests (``test_client_spec.py`` etc.); these
tests cover the wrapper layer's contract: the result is YAML text with
the expected ``kind`` / ``metadata.name``, the patch hook is honoured,
and the ``start_at_epoch`` plumbing reaches the inner client builder.
"""

from __future__ import annotations

import shutil

import pytest
import yaml

from kube_autotuner.benchmark import manifests
from kube_autotuner.experiment import (
    FortioSection,
    IperfSection,
    Patch,
    PatchTarget,
)


def _docs(text: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]


def test_render_iperf_server_returns_yaml_text():
    text = manifests.render_iperf_server(
        node="kmain08",
        ip_family_policy="SingleStack",
        ports=[5201, 5202],
        iperf_args=IperfSection(),
        patches=[],
    )
    assert isinstance(text, str)
    docs = _docs(text)
    kinds = [d.get("kind") for d in docs]
    assert "Deployment" in kinds
    assert "Service" in kinds
    dep = next(d for d in docs if d["kind"] == "Deployment")
    assert dep["metadata"]["name"] == "iperf3-server-kmain08"


def test_render_iperf_client_returns_yaml_text_and_threads_start_at_epoch():
    text_with_barrier = manifests.render_iperf_client(
        source_node="kmain07",
        target_node="kmain08",
        port=5201,
        mode="tcp",
        iperf_args=IperfSection(),
        patches=[],
        start_at_epoch=1_700_000_500,
    )
    assert isinstance(text_with_barrier, str)
    docs = _docs(text_with_barrier)
    job = next(d for d in docs if d["kind"] == "Job")
    assert job["metadata"]["name"] == "iperf3-client-kmain07-p5201"
    # Barrier prologue includes the literal epoch in the rendered shell.
    assert "1700000500" in text_with_barrier

    text_no_barrier = manifests.render_iperf_client(
        source_node="kmain07",
        target_node="kmain08",
        port=5201,
        mode="tcp",
        iperf_args=IperfSection(),
        patches=[],
        start_at_epoch=None,
    )
    # No epoch in the rendered shell when the barrier is disabled.
    assert "1700000500" not in text_no_barrier
    assert "DELTA=" not in text_no_barrier


def test_render_fortio_server_returns_yaml_text():
    text = manifests.render_fortio_server(
        node="kmain08",
        ip_family_policy="SingleStack",
        fortio_args=FortioSection(),
        patches=[],
    )
    assert isinstance(text, str)
    docs = _docs(text)
    dep = next(d for d in docs if d["kind"] == "Deployment")
    assert dep["metadata"]["name"] == "fortio-server-kmain08"


def test_render_fortio_client_saturation_uses_qps_zero():
    text = manifests.render_fortio_client(
        source_node="kmain07",
        target_node="kmain08",
        iteration=2,
        workload="saturation",
        fortio_args=FortioSection(fixed_qps=2500),
        patches=[],
        start_at_epoch=None,
    )
    docs = _docs(text)
    job = next(d for d in docs if d["kind"] == "Job")
    # saturation workload normalises ``_`` to ``-`` for RFC 1123 compliance.
    assert job["metadata"]["name"] == "fortio-client-kmain07-saturation-i2"
    args = job["spec"]["template"]["spec"]["containers"][0]["args"]
    script = args[0]
    assert "-qps 0 " in script or "-qps 0\n" in script
    # The fixed_qps value must NOT be threaded through for saturation.
    assert "2500" not in script


def test_render_fortio_client_fixed_qps_uses_section_value():
    text = manifests.render_fortio_client(
        source_node="kmain07",
        target_node="kmain08",
        iteration=0,
        workload="fixed_qps",
        fortio_args=FortioSection(fixed_qps=2500),
        patches=[],
        start_at_epoch=None,
    )
    docs = _docs(text)
    job = next(d for d in docs if d["kind"] == "Job")
    assert job["metadata"]["name"] == "fortio-client-kmain07-fixed-qps-i0"
    script = job["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "2500" in script


@pytest.mark.skipif(
    shutil.which("kustomize") is None,
    reason="kustomize binary required on PATH",
)
def test_render_iperf_server_honours_patches():
    """A trivial Deployment patch flows through ``apply_patches``."""
    patches = [
        Patch(
            target=PatchTarget(kind="Deployment"),
            patch={"spec": {"replicas": 3}},
        ),
    ]
    text = manifests.render_iperf_server(
        node="kmain08",
        ip_family_policy="SingleStack",
        ports=[5201],
        iperf_args=IperfSection(),
        patches=patches,
    )
    docs = _docs(text)
    dep = next(d for d in docs if d["kind"] == "Deployment")
    assert dep["spec"]["replicas"] == 3
    # The patched output is still parseable YAML carrying the original
    # metadata.name.
    assert dep["metadata"]["name"] == "iperf3-server-kmain08"
