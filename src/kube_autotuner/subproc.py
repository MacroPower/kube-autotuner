"""Sanctioned subprocess entrypoint for the entire package.

Every shell-out — ``kubectl``, ``talosctl``, ``kustomize`` — routes
through :func:`run_tool`. Bare :func:`subprocess.run` calls are banned
in production code so the package has a single choke point for
hygiene: ``capture_output=True``, ``text=True``, pre-allocated error
messages (TRY003 compliance), and explicit ``check`` semantics.
"""

from __future__ import annotations

import subprocess  # noqa: S404 - run_tool is the package's single subprocess entrypoint
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def run_tool(
    binary: str,
    args: Sequence[str],
    *,
    check: bool = False,
    input_: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an external tool and return its completed process.

    The call is always ``capture_output=True, text=True`` so callers get
    decoded ``stdout``/``stderr`` strings without extra plumbing. The
    process is invoked with the argv ``[binary, *args]`` — ``shell=False``
    — so no shell quoting or expansion occurs.

    Args:
        binary: Executable name or absolute path. ``PATH`` resolution
            is delegated to the OS; S607's "partial path" warning is
            accepted here as the intended behaviour.
        args: Arguments passed after ``binary``. Must be an ordered
            sequence of strings; the helper does not flatten nested
            iterables.
        check: When ``True``, raise :class:`subprocess.CalledProcessError`
            on a non-zero exit code. Defaults to ``False`` so callers
            can inspect ``returncode`` and raise a domain-specific error
            (e.g. :class:`kube_autotuner.k8s.client.KubectlError`) with
            richer context.
        input_: Optional string written to the child's ``stdin``.

    Returns:
        The :class:`subprocess.CompletedProcess` with string
        ``stdout`` and ``stderr`` populated. When ``check`` is ``True``
        and the child exits non-zero,
        :class:`subprocess.CalledProcessError` propagates; when
        ``binary`` cannot be resolved on ``PATH``,
        :class:`FileNotFoundError` propagates.
    """
    cmd = [binary, *args]
    return subprocess.run(  # noqa: S603
        cmd,
        input=input_,
        capture_output=True,
        text=True,
        check=check,
    )
