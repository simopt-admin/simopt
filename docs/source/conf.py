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

# Figure out the top-level directory and add it to the system path
# This is necessary to import the simopt package
src_path = Path(__file__).resolve().parent
project_path = src_path.parents[1]
sys.path.insert(0, str(project_path))

# -- Project information -----------------------------------------------------

project = "SimOpt"
copyright = "2025, simopt-admin"  # noqa: A001
author = "simopt-admin"
release = "1.2.2.post0"

# -- General configuration ---------------------------------------------------

master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/Numpy-style docstrings
    "autoapi.extension",  # main AutoAPI extension
]
templates_path = ["_templates"]

# AutoAPI settings
autoapi_type = "python"
autoapi_dirs = [str(project_path / "simopt")]
main_path = str(project_path / "simopt" / "__main__.py")
gui_file_path = str(project_path / "simopt" / "GUI.py")
gui_folder_path = str(project_path / "simopt" / "gui")
autoapi_ignore = [main_path, gui_file_path, gui_folder_path]
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_python_class_content = "both"  # Include docstring from __init__ & class-level
autoapi_options = ["members", "undoc-members", "show-inheritance"]

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
html_static_path = ["_static"]
html_logo = "_static/simopt_logo.png"
html_theme_options = {
    "logo_only": True,  # No name since it is in the logo
    "style_nav_header_background": "#343131",  # Same color as accent color
}

# LaTeX config (if using)
latex_engine = "xelatex"
