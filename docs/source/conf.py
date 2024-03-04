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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'pref_voting'
copyright = '2023, Wes Holliday and Eric Pacuit'
author = 'Wes Holliday and Eric Pacuit'

# The full version, including alpha/beta/rc tags
release = '0.8.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_title = "Preferential Voting Tools"
extensions = [
    'myst_parser',
    'sphinx_copybutton', 
    'sphinx.ext.autodoc', 
    'sphinx.ext.viewcode', 
    'sphinx_exec_code',
    'matplotlib.sphinxext.plot_directive', 
    'sphinx.ext.napoleon'
    ]

plot_formats = ['png', 'pdf']
#autodoc_member_order = 'bysource' 
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    #"fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
   # "strikethrough",
    "substitution",
    "tasklist",
]

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

#def linkcode_resolve(domain, info):#
#    if domain != 'py':
#        return None
#    if not info['module']:
#        return None
#    filename = info['module'].replace('.', '/')
#    return "./%s.py" % filename
