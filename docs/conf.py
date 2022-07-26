# If you have added/deleted files/modules, you will need to run the following sphinx commands from simopt/simopt/docs and push to github
# for those changes to be reflected on readthedocs

# sphinx-apidoc -o . .. -f     # pushing after running this should be enough for readthedocs to be able to generate documentation
# make clean                   # this command and the next one are for building the html locally
# make html




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
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'SimOpt'
copyright = '2021, simopt-admin'
author = 'simopt-admin'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx'
]

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}


autodoc_mock_imports = ['numpy',
                        'scipy',
                        'matplotlib',
                        'pandas',
                        'seaborn'
]

napolean_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '*test*', '*main*']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'default'


latex_engine = 'xelatex'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

#html_static_path = ['_static']
html_static_path = []





from sphinx.ext.apidoc import main
main(["-o",  os.path.abspath('.'), os.path.abspath('..'), "-f"])
