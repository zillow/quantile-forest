# -*- coding: utf-8 -*-
#
# quantile-forest documentation build configuration file.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# -- General configuration ----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinxext_altair.altairplot",
    "sphinxext.directives",
    "sphinxext.gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "quantile-forest"
copyright = f"2022-{datetime.now().year}, Zillow Group"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
from quantile_forest import __version__  # noqa

version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "_templates"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for math equations -----------------------------------------------

# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"

# -- Options for bibtex -------------------------------------------------------

bibtex_bibfiles = ["refs.bib"]

# -- Options for HTML output --------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # "navbar_start": ["navbar-logo", "navbar-project"],
    "navbar_start": ["navbar-logo", "navbar-project"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "primary_sidebar_end": [],
    "logo": {"image_dark": "_static/quantile-forest-logo-white.svg"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/zillow/quantile-forest",
            "icon": "fab fa-github fa-lg",
            "type": "fontawesome",
        },
    ],
}

html_context = {"default_mode": "light"}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "_images"]

# Output file base name for HTML help builder.
html_short_title = "quantile-forest"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/quantile-forest-logo.svg"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

htmlhelp_basename = "quantile-forestdoc"


# Adapted from: http://rackerlabs.github.io/docs-rackspace/tools/rtd-tables.html
# and https://github.com/rtfd/sphinx_rtd_theme/issues/117
def setup(app):
    app.add_css_file("theme_overrides.css")


html_css_files = [
    "css/gallery.css",
]

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": [],
    "releases/changes": [],
    "**": ["sidebar-nav-bs"],
}

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# Generate autosummary even if no references.
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# Hide extra class members.
numpydoc_show_class_members = False

# -- Options for intersphinx --------------------------------------------------

# intersphinx configuration.
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        "index",
        "quantile-forest.tex",
        "quantile-forest Documentation",
        "Zillow Group",
        "manual",
    ),
]

# -- Options for manual page output -------------------------------------------

# If false, no module index is generated.
# latex_domain_indices = True

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        "index",
        "quantile-forest",
        "quantile-forest Documentation",
        ["Zillow Group"],
        1,
    )
]

# If true, show URL addresses after external links.
# man_show_urls = False

# -- Options for copybutton ---------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
