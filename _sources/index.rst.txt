:html_theme.sidebar_secondary.remove:

quantile-forest Documentation
=============================
.. role:: raw-html(raw)
   :format: html

**Version**: |version|

.. rst-class:: lead

   **quantile-forest** is an implementation of scikit-learn compatible quantile regression forests.

   Quantile regression forests (QRF) are a non-parametric, tree-based ensemble method for estimating conditional quantiles, with application to high-dimensional data and uncertainty estimation. The estimators in this package are performant, Cython-optimized QRF implementations that extend the forest estimators available in scikit-learn to estimate conditional quantiles, as described by :cite:t:`2006:meinshausen`. The estimators can estimate arbitrary quantiles at prediction time without retraining and provide methods for out-of-bag estimation, calculating quantile ranks, and computing proximity counts. They are compatible with and can serve as drop-in replacements for the scikit-learn variants.

.. grid:: 1 1 2 2
   :padding: 0 2 3 5
   :gutter: 2 2 3 3
   :class-container: startpage-grid

   .. grid-item-card:: Getting Started
      :link: install
      :link-type: ref
      :link-alt: Getting started

      A guide that provides installation instructions and information on testing the package.

   .. grid-item-card:: User Guide
      :link: user_guide
      :link-type: ref
      :link-alt: User guide

      Check out the User Guide for information on the key concepts behind quantile forests.

   .. grid-item-card:: Examples
      :link: example-gallery
      :link-type: ref
      :link-alt: Examples

      General-purpose, introductory and illustrative examples of using quantile forests.

   .. grid-item-card:: API
      :link: api
      :link-type: ref
      :link-alt: api

      Information on all of the package methods and classes, for when you want just the details.

.. toctree::
   :maxdepth: 1
   :hidden:

   Getting Started <install>
   User Guide <user_guide>
   Examples <gallery/index>
   API <api>

.. _GitHub: http://github.com/zillow/quantile-forest
