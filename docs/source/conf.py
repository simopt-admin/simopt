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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

from sphinx.ext.apidoc import main

simopt_path = Path("..", "simopt").resolve()
project_root = Path("..").resolve()

# Insert paths if they are not already in sys.path
for path in [simopt_path, project_root]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# -- Project information -----------------------------------------------------

project = "SimOpt"
copyright = "2025, simopt-admin"  # noqa: A001
author = "simopt-admin"

# The full version, including alpha/beta/rc tags
release = "1.1.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

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
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*test*", "*main*"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "default"


latex_engine = "xelatex"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]


output_dir = Path().resolve()
source_dir = Path("..").resolve()

main(["-o", str(output_dir), str(source_dir), "-f"])
