"""Post-hoc presentation layer: pareto analysis and HTML rendering.

Layer rule: this subpackage may import any core module
(`models`, `scoring`, `experiment`, `units`, `sysctl.params`, etc.).
**Nothing in core may import from `kube_autotuner.report.*`.**
The CLI's `analyze` command lazy-imports this subpackage from inside
the command body; that is the only sanctioned entry point.

The companion "import ceiling" docstring in `kube_autotuner.scoring`
spells out the inverse direction (scoring is core, must stay
pandas-free, may not reach into this subpackage).
"""

__all__: list[str] = []
