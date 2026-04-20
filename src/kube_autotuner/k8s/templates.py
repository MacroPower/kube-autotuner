"""YAML template rendering helpers.

Templates ship inside the ``kube_autotuner.templates`` resource
package (no ``__init__.py``; resource data only) and are rendered
with :class:`string.Template` so unrecognised ``${...}`` placeholders
are preserved instead of raising ``KeyError``.
"""

from __future__ import annotations

from importlib import resources
from string import Template


def render_template(template_name: str, variables: dict[str, str]) -> str:
    """Render a YAML template from the kube_autotuner templates package.

    Args:
        template_name: Filename inside the ``templates/`` resource
            directory (e.g. ``"lease.yaml"``).
        variables: Mapping of placeholder name to substitution value.

    Returns:
        The template text with ``${VAR}`` placeholders substituted via
        :meth:`string.Template.safe_substitute`. Unknown placeholders
        are left in place rather than raising ``KeyError``.
    """
    templates_pkg = resources.files("kube_autotuner") / "templates"
    template_path = templates_pkg / template_name
    raw = template_path.read_text()
    return Template(raw).safe_substitute(variables)
