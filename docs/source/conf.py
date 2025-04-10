# noqa: D100
# If you have added/deleted files/modules, you will need to run the following sphinx
# commands from simopt/simopt/docs and push to github for those changes to be reflected
# on readthedocs

# pushing after running this should be enough for readthedocs to be able to
# generate documentation:
# `sphinx-apidoc -o . .. -f`

# This command and the next one are for building the html locally
# `make clean`
# `make html`

import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# Get the /src directory
src_path = Path(__file__).resolve().parent
# Use that to get the /docs directory and other /docs/* directories
docs_path = src_path.parent
template_path = docs_path / "_templates"
static_path = docs_path / "_static"
# Get the top level directory of the project and add it to the path
project_path = docs_path.parent
sys.path.insert(0, str(project_path))

# -- Project information -----------------------------------------------------

project = "SimOpt"
copyright = "2025, simopt-admin"  # noqa: A001
author = "simopt-admin"
release = "1.1.1"

# -- General configuration ---------------------------------------------------

master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

templates_path = [str(template_path)]

# Be careful not to exclude important files
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "/test/",
    "/dev_tools/",
    "*main*",
]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = [str(static_path)]

# LaTeX config (if using)
latex_engine = "xelatex"

# -- DO NOT run sphinx-apidoc here ------------------------------------------
# If you want to generate API docs, do it manually or in a separate script.
# For example:
# `sphinx-apidoc -o source ../simopt -f`
#
# Running it here can break builds on Read the Docs due to repeated calls.
