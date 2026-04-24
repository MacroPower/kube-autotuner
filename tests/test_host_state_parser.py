"""Unit tests for the host-state script, parser, and setter-pod rendering."""

from __future__ import annotations

import yaml

from kube_autotuner.k8s.templates import render_template
from kube_autotuner.sysctl.setter import (
    _HOST_STATE_COMMANDS,  # noqa: PLC2701
    _HOST_STATE_SCRIPT,  # noqa: PLC2701
    _parse_host_state_output,  # noqa: PLC2701
)


def _fake_output(**overrides: str) -> str:
    """Return a minimal happy-path stdout, overriding sections with ``overrides``.

    Each kwarg replaces one section's body verbatim.
    """
    defaults = {
        "conntrack_count": "42",
        "conntrack_stats": (
            "cpu=0 found=1 invalid=2 insert=3 insert_failed=0 drop=0 "
            "early_drop=0 error=0 search_restart=7\n"
            "cpu=1 found=4 invalid=1 insert=2 insert_failed=0 drop=0 "
            "early_drop=0 error=0 search_restart=3"
        ),
        "sockstat": (
            "sockets: used 12\n"
            "TCP: inuse 3 orphan 0 tw 5 alloc 4 mem 0\n"
            "UDP: inuse 0 mem 0"
        ),
        "sockstat6": "TCP6: inuse 0\nUDP6: inuse 0",
        "netstat": (
            "TcpExt: TW TWRecycled ListenDrops ListenOverflows DelayedACKs "
            "TCPTimeWaitOverflow TCPOrphanQueued TCPAbortOnData "
            "TCPAbortOnClose TCPKeepAlive\n"
            "TcpExt: 11 0 0 0 99 0 0 0 0 0"
        ),
        "snmp": (
            "Tcp: InSegs OutSegs RetransSegs OutRsts CurrEstab\n"
            "Tcp: 100 200 3 0 7\n"
            "Udp: InDatagrams OutDatagrams RcvbufErrors SndbufErrors\n"
            "Udp: 50 60 0 0"
        ),
        "tcp_metrics": "0",
        "route": "3",
        "arp": "5",
        "meminfo": (
            "MemTotal:       1000000 kB\n"
            "Slab:             12345 kB\n"
            "SReclaimable:      6789 kB\n"
            "SUnreclaim:        5556 kB"
        ),
        "slabinfo": (
            "nf_conntrack     15 20 256 16 1 : tunables 0 0 0 : slabdata 1 1 0"
        ),
        "file_nr": "2048\t0\t65536",
    }
    defaults.update(overrides)
    parts: list[str] = []
    for name, body in defaults.items():
        parts.extend((f"==={name}===", body))
    parts.append("===end===")
    return "\n".join(parts)


def test_parser_happy_path_populates_every_metric():
    """Every configured source contributes at least one metric on clean stdout."""
    metrics, errors = _parse_host_state_output(_fake_output())
    assert errors == []

    # conntrack
    assert metrics["conntrack_count"] == 42
    assert metrics["conntrack_found"] == 5  # 1 + 4 summed across cpus
    assert metrics["conntrack_search_restart"] == 10  # 7 + 3

    # sockstat: keys are flattened as {prefix}{proto}_{field}
    assert metrics["sockstat_tcp_inuse"] == 3
    assert metrics["sockstat_tcp_tw"] == 5
    assert metrics["sockstat_udp_inuse"] == 0
    assert metrics["sockstat6_tcp6_inuse"] == 0

    # netstat / snmp with selected keys only
    assert metrics["netstat_tcpext_TW"] == 11
    assert metrics["netstat_tcpext_DelayedACKs"] == 99
    assert metrics["snmp_tcp_InSegs"] == 100
    assert metrics["snmp_udp_OutDatagrams"] == 60

    # single-value sections
    assert metrics["tcp_metrics_rows"] == 0
    assert metrics["route_rows"] == 3
    assert metrics["arp_rows"] == 5

    # meminfo, stripped of kB suffix
    assert metrics["slab_kb"] == 12345
    assert metrics["sreclaimable_kb"] == 6789
    assert metrics["sunreclaim_kb"] == 5556

    # slabinfo: active_objs then num_objs
    assert metrics["slab_nf_conntrack_active_objs"] == 15
    assert metrics["slab_nf_conntrack_num_objs"] == 20

    # file_nr
    assert metrics["file_nr_allocated"] == 2048
    assert metrics["file_nr_unused"] == 0
    assert metrics["file_nr_max"] == 65536


def test_parser_na_sections_never_produce_sentinel_zero():
    """A source that emitted ``NA`` is absent from metrics and noted in errors."""
    metrics, errors = _parse_host_state_output(
        _fake_output(
            conntrack_count="NA",
            tcp_metrics="NA",
            slabinfo="NA",
        )
    )
    # Absent, NOT metrics["key"] = 0.
    assert "conntrack_count" not in metrics
    assert "tcp_metrics_rows" not in metrics
    assert "slab_nf_conntrack_active_objs" not in metrics

    joined = " | ".join(errors)
    assert "conntrack_count: NA" in joined
    assert "tcp_metrics: NA" in joined
    assert "slabinfo: NA" in joined


def test_parser_missing_section_recorded_as_error():
    """A section the script was expected to emit but didn't lands in errors."""
    # Truncate the faked stdout to drop the file_nr section entirely.
    raw = _fake_output()
    raw = raw.split("===file_nr===", maxsplit=1)[0] + "===end===\n"
    metrics, errors = _parse_host_state_output(raw)
    assert "file_nr_allocated" not in metrics
    assert any("file_nr: section missing" in e for e in errors)


def test_parser_unparseable_int_lands_in_errors():
    """A garbled integer value routes to errors with the offending token."""
    metrics, errors = _parse_host_state_output(
        _fake_output(conntrack_count="not-a-number")
    )
    assert "conntrack_count" not in metrics
    assert any("conntrack_count" in e and "not-a-number" in e for e in errors)


def test_parser_netstat_missing_wanted_key_reported():
    """A wanted ``/proc/net/netstat`` key the kernel didn't emit is flagged."""
    netstat_without_keepalive = (
        "TcpExt: TW TWRecycled ListenDrops ListenOverflows DelayedACKs "
        "TCPTimeWaitOverflow TCPOrphanQueued TCPAbortOnData "
        "TCPAbortOnClose\n"
        "TcpExt: 11 0 0 0 99 0 0 0 0"
    )
    metrics, errors = _parse_host_state_output(
        _fake_output(netstat=netstat_without_keepalive)
    )
    assert "netstat_tcpext_TCPKeepAlive" not in metrics
    assert any("netstat: missing TCPKeepAlive" in e for e in errors)


def test_parser_conntrack_stats_missing_key_reported():
    """A missing ``conntrack -S`` key (e.g. ``search_restart``) is flagged."""
    stats_without_search_restart = (
        "cpu=0 found=1 invalid=2 insert=3 insert_failed=0 drop=0 early_drop=0 error=0"
    )
    metrics, errors = _parse_host_state_output(
        _fake_output(conntrack_stats=stats_without_search_restart)
    )
    assert "conntrack_search_restart" not in metrics
    assert any("conntrack_stats: missing search_restart" in e for e in errors)


def test_parser_meminfo_missing_key_reported():
    """Dropping ``Slab`` from /proc/meminfo should surface a targeted error."""
    meminfo_without_slab = (
        "MemTotal:       1000000 kB\n"
        "SReclaimable:      6789 kB\n"
        "SUnreclaim:        5556 kB"
    )
    metrics, errors = _parse_host_state_output(
        _fake_output(meminfo=meminfo_without_slab)
    )
    assert "slab_kb" not in metrics
    assert any("meminfo: missing Slab" in e for e in errors)


def test_script_constant_contains_each_expected_section_marker():
    """Every section the parser looks up must appear as an ``echo`` in the script."""
    for section in (
        "conntrack_count",
        "conntrack_stats",
        "sockstat",
        "sockstat6",
        "netstat",
        "snmp",
        "tcp_metrics",
        "route",
        "arp",
        "meminfo",
        "slabinfo",
        "file_nr",
    ):
        assert f"echo '==={section}==='" in _HOST_STATE_SCRIPT, section


def test_script_avoids_forbidden_tokens():
    r"""Script must survive Template.safe_substitute + YAML double-quoted scalar.

    ``$NODE`` / ``$POD_NAME`` / ``$SYSCTL_COMMANDS`` would be clobbered
    by the template's own substitutions. ``$$`` would collapse to ``$``.
    ``"`` breaks the YAML double-quoted scalar; ``\`` would trigger YAML
    escape-sequence interpretation.
    """
    for token in ("$NODE", "$POD_NAME", "$SYSCTL_COMMANDS", "$$", '"', "\\"):
        assert token not in _HOST_STATE_SCRIPT, token


def test_script_is_single_line():
    """YAML double-quoted scalars cannot contain literal newlines."""
    assert "\n" not in _HOST_STATE_SCRIPT


def test_commands_tuple_joins_to_script():
    """The tuple-of-commands constant and the joined script stay in sync."""
    assert "; ".join(_HOST_STATE_COMMANDS) == _HOST_STATE_SCRIPT


def test_setter_pod_yaml_round_trips_with_host_state_script():
    """Rendering the setter-pod template with the script must produce valid YAML.

    Confirms the script survives both ``string.Template.safe_substitute``
    and the YAML double-quoted scalar in the template; the parsed pod's
    command payload must equal ``_HOST_STATE_SCRIPT`` verbatim.
    """
    manifest = render_template(
        "sysctl_setter.yaml",
        {
            "NODE": "n1",
            "POD_NAME": "host-snap-n1-baseline",
            "SYSCTL_COMMANDS": _HOST_STATE_SCRIPT,
        },
    )
    parsed = yaml.safe_load(manifest)
    container = parsed["spec"]["containers"][0]
    assert container["command"][:2] == ["/bin/sh", "-c"]
    assert container["command"][2] == _HOST_STATE_SCRIPT
