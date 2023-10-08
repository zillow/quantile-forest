quantile-forest Documentation
============================================

quantile-forest is an implementation of scikit-learn compatible quantile regression forests.

Quantile regression forests (QRF) are a non-parametric, tree-based ensemble method for estimating conditional quantiles, with application to high-dimensional data and uncertainty estimation. The estimators in this package are performant, Cython-optimized QRF implementations that extend the forest estimators available in scikit-learn to estimate conditional quantiles, as described by :cite:t:`2006:meinshausen`. The estimators can estimate arbitrary quantiles at prediction time without retraining and provide methods for out-of-bag estimation, calculating quantile ranks, and computing proximity counts. They are compatible with and can serve as drop-in replacements for the scikit-learn variants.

.. toctree::
   :maxdepth: 2
   :hidden:

   install

.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide

.. toctree::
   :maxdepth: 2
   :hidden:

   api

.. toctree::
   :maxdepth: 2
   :hidden:

   auto_examples/index

`Getting Started <install.html>`_
-------------------------------------

Information to install and test the package.

`User Guide <user_guide.html>`_
-------------------------------

The main documentation. This provides information and background on the key concepts of quantile forests.

`API Reference <api.html>`_
-------------------------------

The API of all functions and classes, as given in the doctring.

`Examples <auto_examples/index.html>`_
--------------------------------------

General-purpose and introductory examples.
