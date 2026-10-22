"""Sphinx configuration for fitwise."""

import fitwise

project = "fitwise"
author = "Matthias Baer"
copyright = "2026, Matthias Baer"
release = fitwise.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "**/*.ipynb", "sg_execution_times.rst"]

myst_enable_extensions = ["colon_fence", "dollarmath"]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "download_all_examples": False,
}

html_theme = "pydata_sphinx_theme"
html_title = "fitwise"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/maroba/fitwise",
    "navigation_with_keys": False,
}
