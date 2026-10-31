.. _user-guide-intro:

Introduction
------------

Random forests have proven to be very popular and powerful for regression and classification. For regression, random forests give an accurate approximation of the conditional mean of a response variable. That is, if we let :math:`Y` be a real-valued response variable and :math:`X` a covariate or predictor variable, they estimate :math:`E(Y | X)`, which can be interpreted as the expected value of the output :math:`Y` given the input :math:`X`.

However random forests provide information about the full conditional distribution of the response variable, not only about the conditional mean. Quantile regression forests, a generalization of random forests, can be used to infer conditional quantiles. That is, they return :math:`y` at :math:`q` for which :math:`F(Y=y|X) = q`, where :math:`q` is the quantile.

The quantiles give more complete information about the distribution of :math:`Y` as a function of the predictor variable :math:`X` than the conditional mean alone. They can be useful, for example, to build prediction intervals or to perform outlier detection in a high-dimensional dataset.

In practice, the empirical estimation of quantiles can be calculated in several ways. By default, this package calculates the quantile :math:`q` as the smallest training response such that a fraction (at least) :math:`q` of the (weighted) training samples are at or below it, averaged with the next response if exactly that fraction are, so that the median is the usual median. This is NumPy's `averaged_inverted_cdf` method, and how scikit-learn calculates weighted quantiles. Other methods can be specified with the `interpolation` parameter of the `predict` method. With `interpolation="linear"`, the default before version 1.5, a quantile is calculated from the input rank :math:`x` such that :math:`x = (N + 1 - 2C)q + C`, where :math:`N` is the number of samples and :math:`C = 1` is a constant (degree of freedom). For weighted samples, :math:`N` is their total weight instead, with the weights scaled so that a sample of average weight counts as one sample.

Quantile Regression Forests
---------------------------

A standard decision tree can be extended in a straightforward way to estimate conditional quantiles. When a decision tree is fit, rather than storing only the sufficient statistics of the response variable at the leaf node, such as the mean and variance, all of the response values can be stored with the leaf node. At prediction time, these values can then be used to calculate empirical quantile estimates.

The quantile-based approach can be extended to random forests. To estimate :math:`F(Y=y|x) = q`, each response value in the training set is given a weight or frequency. Formally, the weight or frequency given to the :math:`j`\th training sample, :math:`y_j`, while estimating the quantile is

.. math::

  \frac{1}{T} \sum_{t=1}^{T} \frac{\mathbb{1}(y_j \in L(x))}{\sum_{i=1}^N \mathbb{1}(y_i \in L(x))},

where :math:`L(x)` denotes the leaf that :math:`x` falls into.

Informally, this means that given a new unknown sample, we first find the leaf that it falls into for each tree in the ensemble. Each training sample :math:`y_j` that falls into the same leaf as the new sample is given a weight that equals the fraction of samples in the leaf. Each :math:`y_j` that does not fall into the same leaf as the new sample is given a weight or frequency of zero. The weights or frequencies for each :math:`y_j` are then summed or aggregated across all of the trees in the ensemble. This information can then be used to calculate the empirical quantile estimates.

This approach was first proposed by :cite:t:`2006:meinshausen`. This package uses these weights if each leaf keeps all of its training samples (`max_samples_leaf=None`) and `weighted_leaves=True` (see :ref:`user-guide-fit-predict`). By default, each leaf keeps one randomly selected sample (`max_samples_leaf=1`), so that each tree contributes one response value, and if leaves keep more samples, each sample in a leaf counts once, whatever the size of the leaf (`weighted_leaves=False`).

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   self
   fit_predict
   quantile_ranks
   proximities
