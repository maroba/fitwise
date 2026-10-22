# 4. Documentation with Sphinx and sphinx-gallery

*Status: accepted · 2026-10-22*

**Context.** The documentation relies on runnable examples that render their plots. Users
expect it to link into the NumPy, SciPy and lmfit docs.

**Decision.** The documentation uses Sphinx with `pydata-sphinx-theme`, MyST Markdown,
`sphinx-gallery` for the examples (`examples/plot_*.py`) and intersphinx links to the
dependencies.

**Consequences.** The look and tooling match the scientific Python ecosystem. Examples run
during every docs build, so they cannot silently go stale. The build is slower than a static
Markdown site.
