"""Quantile regression forest of trees-based ensemble methods.

The module structure is the following:

- The ``BaseForestQuantileRegressor`` base class implements a common ``fit``
  method for all the estimators in the module. The ``fit`` method of the base
  class calls the ``fit`` method of the ``ForestRegressor`` and creates a
  quantile forest object that records the leaf node membership of all samples
  of the training set.

- The ``RandomForestQuantileRegressor`` derived class provides the user with a
  concrete implementation of the quantile regression forest ensemble method
  that extends the classical ``RandomForestRegressor`` as the estimator
  implementation.

- The ``ExtraTreesQuantileRegressor`` derived class provides the user with a
  concrete implementation of the quantile regression forest ensemble method
  that extends the extremely randomized trees ``ExtraTreesRegressor`` as the
  estimator implementation.

Single- and multi-output problems are both handled.
"""

import random
import warnings
from abc import abstractmethod
from math import ceil
from numbers import Integral, Real
from warnings import warn

import joblib
import numpy as np
import sklearn
from sklearn.ensemble._forest import (
    ForestRegressor,
    _generate_sample_indices,
    _get_n_samples_bootstrap,
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.tree._tree import DTYPE
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_is_fitted

try:
    from sklearn.utils.validation import validate_data
except ImportError:
    validate_data = None

from ._quantile_forest_fast import QuantileForest
from ._utils import generate_unsampled_indices, group_indices_by_value, map_indices_to_leaves

sklearn_version = parse_version(sklearn.__version__)


class BaseForestQuantileRegressor(ForestRegressor):
    """Base class for quantile regression forests.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    _parameter_constraints: dict = {
        **ForestRegressor._parameter_constraints,
        **DecisionTreeRegressor._parameter_constraints,
        # Maximum number of samples to store per tree leaf.
        "max_samples_leaf": [
            None,
            Interval(RealNotInt, 0, 1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
    }
    _parameter_constraints.pop("splitter")

    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
        max_samples_leaf=1,
    ):
        """Initialize base quantile forest regressor."""
        init_dict = {
            "estimator": estimator,
            "n_estimators": n_estimators,
            "estimator_params": estimator_params,
            "bootstrap": bootstrap,
            "oob_score": oob_score,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "warm_start": warm_start,
            "max_samples": max_samples,
        }
        super().__init__(**init_dict)

    def fit(self, X, y, sample_weight=None, sparse_pickle=False):
        """Build a forest from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        sparse_pickle : bool, default=False
            Pickle the underlying data structure using a SciPy sparse matrix.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()

        if self.monotonic_cst is not None:
            if (
                not isinstance(self.max_samples_leaf, (Integral, np.integer))
                or self.max_samples_leaf != 1
            ):
                raise ValueError(
                    "Monotonicity constraints are not supported with multiple values per leaf. "
                    "To apply monotonicity constraints, set `max_samples_leaf=1`."
                )

        super(BaseForestQuantileRegressor, self).fit(X, y, sample_weight=sample_weight)

        validation_params = {
            "X": X,
            "y": y,
            "multi_output": True,
            "accept_sparse": "csc",
            "dtype": DTYPE,
            (
                "force_all_finite"
                if sklearn_version < parse_version("1.6.dev0")
                else "ensure_all_finite"
            ): True,
        }
        if validate_data is None:
            X, y = self._validate_data(**validation_params)
        else:
            X, y = validate_data(self, **validation_params)

        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        # Sort the target values in ascending order.
        # Use sorter to maintain mapping to original order.
        sorter = np.argsort(y, axis=0)
        y_sorted = np.empty_like(y)
        for i in range(y.shape[1]):
            y_sorted[:, i] = y[sorter[:, i], i]

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[sorter]

        # Get map of tree leaf nodes to training indices.
        y_train_leaves = self._get_y_train_leaves(
            X, y_sorted, sorter=sorter, sample_weight=sample_weight
        )

        # Get map of tree leaf nodes to target value bounds (for monotonicity constraints).
        y_bound_leaves = self._get_y_bound_leaves(y_sorted, y_train_leaves)

        # Create quantile forest object.
        self.forest_ = QuantileForest(
            y_sorted.astype(np.float64).T,
            y_train_leaves,
            y_bound_leaves,
            sparse_pickle=sparse_pickle,
        )

        self.sorter_ = sorter
        self.n_train_samples_ = len(y_sorted)
        self.X_train_hash_ = joblib.hash(X)
        self.unsampled_indices_ = None

        return self

    @staticmethod
    def _get_y_train_leaves_slice(
        bootstrap_indices,
        X_leaves_bootstrap,
        sample_weight,
        leaf_subsample,
        max_node_count,
        max_samples_leaf,
        random_state,
    ):
        """Return a mapping of training sample indices to a tree's leaf nodes.

        Parameters
        ----------
        bootstrap_indices : array-like of shape (n_samples, n_outputs)
            Bootstrap indices of training samples.

        X_leaves_bootstrap : array-like of shape (n_samples,)
            Leaf node indices of the bootstrap training samples.

        sample_weight : array-like of shape (n_samples, n_outputs), \
                default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. For each output,
            the ordering of the weights correspond to the sorted samples.

        leaf_subsample : bool
            Subsample leaf nodes. If True, leaves are randomly sampled to size
            `max_samples_leaf`.

        max_node_count : int
            Maximum number of leaf nodes across all trees.

        max_samples_leaf : int
            Maximum number of samples per leaf node.

        random_state : int, RandomState instance
            Controls the sampling of the training indices at each leaf node.

        Returns
        -------
        y_train_leaves_slice : array-like of shape \
                (n_leaves, n_outputs, n_indices)
            Mapping of training sample indices to tree's leaf nodes. Nodes with
            no samples (e.g., internal nodes) are empty. Internal nodes are
            included so that leaf node indices match their ``est.apply``
            outputs. Each node list is padded to equal length with 0s.
        """
        n_outputs = bootstrap_indices.shape[1]

        if sample_weight is not None:
            sample_weight = np.squeeze(sample_weight)

        shape = (max_node_count, n_outputs, max_samples_leaf)
        y_train_leaves_slice = np.zeros(shape, dtype=np.int64)

        # Group training indices by leaf node.
        leaf_indices, leaf_values_list = group_indices_by_value(X_leaves_bootstrap)

        if leaf_subsample or sample_weight is not None:
            if leaf_subsample:
                random.seed(random_state)

            for j in range(len(leaf_values_list)):
                if sample_weight is not None:
                    # Filter leaf samples with zero weight.
                    weight_mask = sample_weight[leaf_values_list[j] - 1] > 0
                    leaf_values_list[j] = leaf_values_list[j][weight_mask]
                if leaf_subsample:
                    # Sample leaf to length `max_samples_leaf`.
                    if len(leaf_values_list[j]) > max_samples_leaf:
                        random.shuffle(leaf_values_list[j])  # to ensure random sampling
                        leaf_values_list[j] = leaf_values_list[j][:max_samples_leaf]
                if len(leaf_values_list[j]) == 0:
                    leaf_values_list[j] = [0]

        # Map each leaf node to its list of training indices.
        if max_samples_leaf == 1:  # optimize for single-sample-per-leaf performance
            y_indices = bootstrap_indices[leaf_values_list].reshape(-1, 1, n_outputs)
            for j in range(n_outputs):
                y_train_leaves_slice[leaf_indices, j, 0] = y_indices[:, 0, j]
        else:  # get mapping for arbitrary leaf sizes
            y_train_leaves_slice = map_indices_to_leaves(
                y_train_leaves_slice=y_train_leaves_slice,
                bootstrap_indices=bootstrap_indices,
                leaf_indices=leaf_indices,
                leaf_values_list=leaf_values_list,
            )

        return y_train_leaves_slice

    def _get_y_train_leaves(self, X, y, sorter=None, sample_weight=None):
        """Return a mapping of each leaf node to its list of training indices.

        The ``apply`` function is used on the ``X`` values to obtain the leaf
        indices for the appropriate training indices, as sorted by ``sorter``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples, n_outputs)
            The target outputs for each y value.

        sorter : array-like of shape (n_samples, n_outputs), default=None
            The indices that would sort the target values in ascending order.
            Used to associate ``est.apply`` outputs with sorted target values.

        sample_weight : array-like of shape (n_samples, n_outputs), \
                default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. For each output,
            the ordering of the weights correspond to the sorted samples.

        Returns
        -------
        y_train_leaves : array-like of shape \
                (n_estimators, n_leaves, n_outputs, n_indices)
            List of trees, each with a list of nodes, each with a list of
            indices of the training samples residing at that node. Nodes with
            no samples (e.g., internal nodes) are empty. Internal nodes are
            included so that leaf node indices match their ``est.apply``
            outputs. Each node list is padded to equal length with 0s.
        """
        n_samples = X.shape[0]
        n_outputs = y.shape[-1]

        if isinstance(self.max_samples_leaf, (Integral, np.integer)):
            max_samples_leaf = self.max_samples_leaf
            leaf_subsample = True
        elif isinstance(self.max_samples_leaf, Real):
            max_samples_leaf = int(ceil(self.max_samples_leaf * n_samples))
            leaf_subsample = True
        elif self.max_samples_leaf is None:
            max_samples_leaf = self.max_samples_leaf
            leaf_subsample = False
        else:
            raise ValueError(
                "`max_samples_leaf` must be of integer, float, or None type, "
                f"got {type(self.max_samples_leaf)}."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            X_leaves = self.apply(X)

        if self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.max_samples)

        shape = (n_samples if not self.bootstrap else n_samples_bootstrap, self.n_estimators)
        bootstrap_indices = np.empty(shape, dtype=np.int64)
        X_leaves_bootstrap = np.empty(shape, dtype=np.int64)
        for i, estimator in enumerate(self.estimators_):
            # Get bootstrap indices.
            if self.bootstrap:
                bootstrap_indices[:, i] = _generate_sample_indices(
                    estimator.random_state, n_samples, n_samples_bootstrap
                )
            else:
                bootstrap_indices[:, i] = np.arange(n_samples)

            # Get leaf node indices of bootstrap training samples.
            X_leaves_bootstrap[:, i] = X_leaves[bootstrap_indices[:, i], i]

        if sorter is not None:
            # Reassign bootstrap indices to account for target sorting.
            bootstrap_indices = np.argsort(sorter, axis=0)[bootstrap_indices]

        bootstrap_indices = bootstrap_indices.reshape(-1, self.n_estimators, n_outputs)
        bootstrap_indices += 1  # for sparse matrix (0s as empty)

        # Get the maximum number of nodes (internal + leaves) across trees.
        # Get the maximum number of samples per leaf across trees (if needed).
        max_node_count = 0
        max_samples_leaf = 0 if not leaf_subsample else max_samples_leaf
        for i, estimator in enumerate(self.estimators_):
            node_count = estimator.tree_.node_count
            max_node_count = max(max_node_count, node_count)
            if not leaf_subsample:
                sample_count = np.max(np.bincount(X_leaves_bootstrap[:, i]))
                max_samples_leaf = max(max_samples_leaf, sample_count)

        y_train_leaves = [
            self._get_y_train_leaves_slice(
                bootstrap_indices[:, i],
                X_leaves_bootstrap[:, i],
                sample_weight,
                leaf_subsample,
                max_node_count,
                max_samples_leaf,
                estimator.random_state,
            )
            for i, estimator in enumerate(self.estimators_)
        ]

        return np.array(y_train_leaves)

    def _get_y_bound_leaves(self, y, y_train_leaves):
        """Return the bounds for target values for each leaf node.

        The target value bounds are used to enforce monotonicity constraints.

        Parameters
        ----------
        y : array-like of shape (n_samples, n_outputs)
            The target outputs for each y value.

        y_train_leaves : array-like of shape \
                (n_estimators, n_leaves, n_outputs, n_indices)
            List of trees, each with a list of nodes, each with a list of
            indices of the training samples residing at that node. Nodes with
            no samples (e.g., internal nodes) are empty. Internal nodes are
            included so that leaf node indices match their ``est.apply``
            outputs. Each node list is padded to equal length with 0s.

        Returns
        -------
        y_bound_leaves : array-like of shape (n_estimators, n_leaves, 2)
            Minimum and maximum bounds for target values for each leaf node.
            Used to enforce monotonicity constraints. None if no constraints.
        """
        if self.monotonic_cst is None:
            return None

        y_bound_leaves = np.full((*y_train_leaves.shape[:2], 2), [-np.inf, np.inf])

        for i, estimator in enumerate(self.estimators_):
            tree = estimator.tree_

            num_values = np.zeros(tree.node_count)
            sum_values = np.zeros(tree.node_count)

            # Iterate over all nodes in reverse order (leaves to root).
            for node_idx in range(tree.node_count - 1, -1, -1):
                if tree.children_left[node_idx] == tree.children_right[node_idx]:  # leaf node
                    # Populate the leaf nodes.
                    leaf_indices = y_train_leaves[i, node_idx]
                    leaf_indices = leaf_indices[leaf_indices != 0]
                    if leaf_indices.size > 0:
                        leaf_targets = y[leaf_indices - 1]
                        num_values[node_idx] = leaf_indices.size
                        sum_values[node_idx] = leaf_targets.sum()
                else:  # non-leaf node
                    # Populate the non-leaf nodes based on the children.
                    left_child = tree.children_left[node_idx]
                    right_child = tree.children_right[node_idx]
                    num_values[node_idx] = num_values[left_child] + num_values[right_child]
                    sum_values[node_idx] = sum_values[left_child] + sum_values[right_child]

            # Traverse from root to leaves to enforce monotonicity.
            stack = [(0, -np.inf, np.inf)]  # start with root node (node 0)

            while stack:
                node_idx, min_bound, max_bound = stack.pop()
                if tree.children_left[node_idx] == tree.children_right[node_idx]:  # leaf node
                    # The bounds have already been calculated in the leaf.
                    y_bound_leaves[i, node_idx] = [min_bound, max_bound]
                else:  # non-leaf node
                    feature_idx = tree.feature[node_idx]

                    left_child = tree.children_left[node_idx]
                    right_child = tree.children_right[node_idx]

                    if self.monotonic_cst[feature_idx] == 0:
                        stack.append((left_child, min_bound, max_bound))
                        stack.append((right_child, min_bound, max_bound))
                    else:
                        # Calculate midpoint that respects the current node's bounds.
                        left_avg = sum_values[left_child] / (2 * num_values[left_child])
                        right_avg = sum_values[right_child] / (2 * num_values[right_child])
                        mid = max(min(left_avg + right_avg, max_bound), min_bound)

                        if self.monotonic_cst[feature_idx] == 1:  # increasing monotonicity
                            stack.append((left_child, min_bound, mid))
                            stack.append((right_child, mid, max_bound))
                        elif self.monotonic_cst[feature_idx] == -1:  # decreasing monotonicity
                            stack.append((left_child, mid, max_bound))
                            stack.append((right_child, min_bound, mid))

        return y_bound_leaves

    def _oob_samples(self, X, indices=None, duplicates=None):
        """Generate out-of-bag (OOB) samples for each base estimator.

        Only generates leaf indices for samples that were excluded from the
        bootstrapping process for each base estimator. If ``indices`` is None,
        assumes that ``X`` is the same length and order as the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        indices : list, default=None
            List of training indices that correspond to X indices. An index of
            -1 can be used to specify rows omitted from the training set. By
            default, assumes all X indices correspond to all training indices.

        duplicates : list, default=None
            List of sets of functionally identical indices.

        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators)
            Prediction leaves for OOB samples. Non-OOB samples may have
            uninitialized (arbitrary) data.

        X_indices : array-like of shape (n_samples, n_estimators)
            Mask for OOB samples. 1 if OOB sample, 0 otherwise.
        """
        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)

        if indices is None:
            if n_samples != self.n_train_samples_:
                raise ValueError(
                    "If `indices` are None, OOB samples must be same length as training samples."
                )
            elif joblib.hash(X) != self.X_train_hash_:
                warn("OOB samples are not identical to training samples.")

        if indices is not None and n_samples != len(indices):
            raise ValueError("If `indices` are not None, X and indices must be the same length.")

        X_leaves = np.empty((n_samples, n_estimators), dtype=np.intp)
        X_indices = np.zeros((n_samples, n_estimators), dtype=np.uint8)
        for i, estimator in enumerate(self.estimators_):
            # Get the indices excluded from the bootstrapping process.
            if self.unsampled_indices_ is None:
                unsampled_indices = self._get_unsampled_indices(estimator, duplicates=duplicates)
            else:
                # Avoid generating unsampled indices if they are precomputed.
                unsampled_indices = np.asarray(self.unsampled_indices_[i])

            if unsampled_indices.size > 0 and indices is not None:
                # Select only those unsampled indices specified by `indices`.
                # Token index (-1) denotes indices that are always unsampled.
                if len(indices) == 1:  # optimize for single-row performance
                    unsampled_indices = [
                        x_i for x_i, x in enumerate(indices) if x in unsampled_indices or x == -1
                    ]
                    unsampled_indices = np.array(unsampled_indices)
                else:
                    unsampled_mask = np.isin(indices, unsampled_indices)
                    unsampled_mask = unsampled_mask | np.equal(indices, -1)
                    unsampled_indices = np.arange(len(indices))[unsampled_mask]

            if unsampled_indices.size > 0:
                # Check data.
                X_unsampled = self._validate_X_predict(X[unsampled_indices])

                # Get leaf node indices for unsampled training samples.
                X_leaves[unsampled_indices, i] = estimator.apply(X_unsampled)

                # Set indicator value for unsampled training samples.
                X_indices[unsampled_indices, i] = 1

        return X_leaves, X_indices

    def _get_unsampled_indices(self, estimator, duplicates=None):
        """Get the unsampled indices for a base estimator.

        Parameters
        ----------
        estimator : object
            Base estimator.

        duplicates : list of lists, default=None
            List of sets of functionally identical indices.

        Returns
        -------
        unsampled_indices : array of shape (n_unsampled,)
            Unsampled indices.
        """
        if not self.bootstrap:
            warn("Unsampled indices only exist if bootstrap=True.")
            return np.array([])
        n_train_samples = self.n_train_samples_
        n_samples_bootstrap = _get_n_samples_bootstrap(n_train_samples, self.max_samples)
        sample_indices = _generate_sample_indices(
            estimator.random_state, n_train_samples, n_samples_bootstrap
        )
        unsampled_indices = generate_unsampled_indices(
            sample_indices,
            n_train_samples,
            duplicates=[] if duplicates is None else duplicates,
        )
        return np.asarray(unsampled_indices)

    def predict(
        self,
        X,
        quantiles=None,
        interpolation="linear",
        weighted_quantile=True,
        weighted_leaves=False,
        aggregate_leaves_first=True,
        oob_score=False,
        indices=None,
        duplicates=None,
    ):
        """Predict quantiles for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        quantiles : float, list, or "mean", default=None
            The quantile or list of quantiles that the model tries to predict.
            Each quantile must be strictly between 0 and 1. If "mean", the
            model predicts the mean. If None, the model uses the value of
            `default_quantiles`.

        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}, \
                default="linear"
            Specifies the interpolation method to use when the desired
            quantile lies between two data points ``i < j``:

            - If "linear", then ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i`` and
              ``j``.
            - If "lower", then ``i``.
            - If "higher", then ``j``.
            - If "nearest", then ``i`` or ``j``, whichever is nearest.
            - If "midpoint", then ``(i + j) / 2``.

            .. note::
                When `max_samples_leaf=1`, the specific interpolation
                options will depend on those available in the NumPy
                `quantile` and `nanquantile` methods.

        weighted_quantile : bool, default=True
            Calculate a weighted quantile. Weighted quantiles are computed by
            assigning weights to each training sample, while unweighted
            quantiles are computed by aggregating sibling samples. When the
            number of training samples relative to siblings is small, weighted
            quantiles can be more efficient to compute than unweighted ones.

        weighted_leaves : bool, default=False
            Weight samples inversely to the size of their leaf node.
            Only used if `weighted_quantile=True` and `max_samples_leaf!=1`.

        aggregate_leaves_first : bool, default=True
            Calculate predictions using aggregated leaf values. If True, a
            single prediction is calculated over the aggregated leaf values.
            If False, a prediction is calculated for each leaf and aggregated.

        oob_score : bool, default=False
            Only use out-of-bag (OOB) samples to predict quantiles.

        Other Parameters
        ----------------
        indices : list, default=None
            List of training indices that correspond to X indices. An index of
            -1 can be used to specify rows omitted from the training set. By
            default, assumes all X indices correspond to all training indices.
            Only used if `oob_score=True`.

        duplicates : list of lists, default=None
            List of sets of functionally identical indices.
            Only used if `oob_score=True`.

        Returns
        -------
        y_pred : array of shape (n_samples,) or (n_samples, n_quantiles) or \
                (n_samples, n_outputs, n_quantiles)
            If quantiles is set to 'mean', then return ``E(Y | X)``. Else, for
            all quantiles, return ``y`` at ``q`` for which ``F(Y=y|x) = q``,
            where ``q`` is the quantile.
        """
        check_is_fitted(self)
        # Check data.
        X = self._validate_X_predict(X)

        if quantiles is None:
            quantiles = ["mean"] if self.default_quantiles is None else self.default_quantiles
        if not isinstance(quantiles, list):
            quantiles = [quantiles]

        if quantiles == ["mean"]:
            quantiles = [-1]
        else:
            for q in quantiles:
                if not isinstance(q, (float, int)) or (q < 0 or q > 1):
                    raise ValueError(f"Quantiles must be in the range [0, 1], got {q}.")

        if not isinstance(interpolation, (bytes, bytearray)):
            interpolation = interpolation.encode()

        if oob_score:
            if not self.bootstrap:
                raise ValueError("Out-of-bag estimation only available if bootstrap=True.")

            X_leaves, X_indices = self._oob_samples(X, indices, duplicates)

            if (X_indices.sum(axis=1) == 0).any():
                warn(
                    "Some inputs do not have OOB scores. "
                    "This probably means too few trees were used "
                    "to compute any reliable OOB estimates."
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                X_leaves = self.apply(X)
            X_indices = None

        if self.max_samples_leaf == 1:  # optimize for single-sample-per-leaf performance
            y_train_leaves = np.asarray(self.forest_.y_train_leaves)
            y_bound_leaves = np.asarray(self.forest_.y_bound_leaves)
            y_train = np.asarray(self.forest_.y_train).T
            y_pred = np.empty((len(X), y_train.shape[1], len(quantiles)))
            for output in range(y_train.shape[1]):
                leaf_vals = np.empty((len(X), self.n_estimators))
                for tree in range(self.n_estimators):
                    if X_indices is None:  # in-bag scoring
                        leaves = X_leaves[:, tree]
                        train_indices = y_train_leaves[tree, leaves, output, 0]
                        leaf_vals[:, tree] = y_train[train_indices - 1, output]
                        if self.monotonic_cst is not None:
                            clip_min = y_bound_leaves[tree, leaves, 0]
                            clip_max = y_bound_leaves[tree, leaves, 1]
                            leaf_vals[:, tree] = np.clip(leaf_vals[:, tree], clip_min, clip_max)
                    else:  # out-of-bag scoring
                        idx_mask = X_indices[:, tree] == 1
                        leaves = X_leaves[idx_mask, tree]
                        train_indices = np.zeros(len(X), dtype=int)
                        train_indices[idx_mask] = y_train_leaves[tree, leaves, output, 0]
                        leaf_vals[:, tree] = np.nan
                        leaf_vals[idx_mask, tree] = y_train[train_indices[idx_mask] - 1, output]
                        if self.monotonic_cst is not None:
                            clip_min, clip_max = np.full(len(X), -np.inf), np.full(len(X), np.inf)
                            clip_min[idx_mask] = y_bound_leaves[tree, leaves, 0]
                            clip_max[idx_mask] = y_bound_leaves[tree, leaves, 1]
                            leaf_vals[:, tree] = np.clip(leaf_vals[:, tree], clip_min, clip_max)

                if len(quantiles) == 1 and quantiles[0] == -1:  # calculate mean
                    func = np.mean if X_indices is None else np.nanmean
                    y_pred[:, output, :] = np.expand_dims(func(leaf_vals, axis=1), axis=1)
                else:  # calculate quantiles
                    func = np.quantile if X_indices is None else np.nanquantile
                    method = interpolation.decode()
                    y_pred[:, output, :] = func(leaf_vals, quantiles, method=method, axis=1).T
        else:  # get predictions for arbitrary leaf sizes
            y_pred = self.forest_.predict(
                quantiles,
                X_leaves,
                X_indices,
                interpolation,
                weighted_quantile,
                weighted_leaves,
                aggregate_leaves_first,
            )

        if y_pred.shape[2] == 1:
            y_pred = np.squeeze(y_pred, axis=2)

        if y_pred.shape[1] == 1:
            y_pred = np.squeeze(y_pred, axis=1)

        return y_pred

    def quantile_ranks(
        self,
        X,
        y,
        kind="rank",
        aggregate_leaves_first=True,
        oob_score=False,
        indices=None,
        duplicates=None,
    ):
        """Return quantile ranks for X with scores y.

        A quantile rank of, for example, 0.8 means that 80% of the scores in
        the frequency distribution of the inputs are below the given score.

        In the case of gaps or ties, the exact definition depends on the
        optional keyword `kind`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values for which to calculate ranks.

        kind : {"rank", "weak", "strict", "mean"}, default="rank"
            Specifies the interpretation of the resulting score:

            - If "rank", then average percentage ranking of score. If multiple
              matches, average the percentage rankings of all matching scores.
            - If "weak", then only values that are less than or equal to the
              provided score are counted. Corresponds to the definition of a
              cumulative distribution function.
            - If "strict", then similar to "weak", except that only values
              that are strictly less than the provided score are counted.
            - If "mean", then the average of the "weak" and "strict" scores.

        aggregate_leaves_first : bool, default=True
            Calculate quantile ranks using aggregated leaf values. If True, a
            single rank is calculated over the aggregated leaf values. If
            False, a rank is calculated for each leaf and aggregated.

        oob_score : bool, default=False
            Only use out-of-bag (OOB) samples to predict quantile ranks.

        Other Parameters
        ----------------
        indices : list, default=None
            List of training indices that correspond to X indices. An index of
            -1 can be used to specify rows omitted from the training set. By
            default, assumes all X indices correspond to all training indices.
            Only used if `oob_score=True`.

        duplicates : list of lists, default=None
            List of sets of functionally identical indices.
            Only used if `oob_score=True`.

        Returns
        -------
        y_ranks : array of shape (n_samples,) or (n_samples, n_outputs)
            Quantile ranks in range [0, 1].
        """
        check_is_fitted(self)

        validation_params = {
            "X": X,
            "y": y,
            "multi_output": True,
            "accept_sparse": "csc",
            "dtype": DTYPE,
            (
                "force_all_finite"
                if sklearn_version < parse_version("1.6.dev0")
                else "ensure_all_finite"
            ): False,
        }
        if validate_data is None:
            X, y = self._validate_data(**validation_params)
        else:
            X, y = validate_data(self, **validation_params)

        if not isinstance(kind, (bytes, bytearray)):
            kind = kind.encode()

        if oob_score:
            if not self.bootstrap:
                raise ValueError("Out-of-bag estimation only available if bootstrap=True.")

            X_leaves, X_indices = self._oob_samples(X, indices, duplicates)

            if (X_indices.sum(axis=1) == 0).any():
                warn(
                    "Some inputs do not have OOB scores. "
                    "This probably means too few trees were used "
                    "to compute any reliable OOB estimates."
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                X_leaves = self.apply(X)
            X_indices = None

        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        y_ranks = self.forest_.quantile_ranks(
            y.astype(np.float64),
            X_leaves,
            X_indices,
            kind,
            aggregate_leaves_first,
        )

        if y_ranks.shape[1] == 1:
            y_ranks = np.squeeze(y_ranks, axis=1)

        return y_ranks

    def proximity_counts(
        self,
        X,
        max_proximities=None,
        return_sorted=True,
        oob_score=False,
        indices=None,
        duplicates=None,
    ):
        """Return training proximity counts for input samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        max_proximities : int, default=None
            Maximum number of proximities to return for each scoring sample,
            prioritized by proximity count. By default, return all proximity
            counts for each sample.

        return_sorted : bool, default=True
            For each sample, sort its proximity counts in descending order.
            If False, the proximities will be returned in arbitrary order.

        oob_score : bool, default=False
            Only use out-of-bag (OOB) samples to generate proximity counts.

        Other Parameters
        ----------------
        indices : list, default=None
            List of training indices that correspond to X indices. An index of
            -1 can be used to specify rows omitted from the training set. By
            default, assumes all X indices correspond to all training indices.
            Only used if `oob_score=True`.

        duplicates : list of lists, default=None
            List of sets of functionally identical indices.
            Only used if `oob_score=True`.

        Returns
        -------
        proximities : list of tuples, length = n_samples
            List of tuples mapping training sample indices to proximity counts.

        Notes
        -----
        For details on the calculation and use of random forest proximities:
            - https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#prox
        """
        check_is_fitted(self)
        # Check data.
        X = self._validate_X_predict(X)

        if max_proximities is None:
            max_proximities = 0
        elif max_proximities < 1:
            raise ValueError(
                "`max_proximities` must be larger than or equal to 1 if not"
                f" None, got {max_proximities}."
            )
        elif not isinstance(max_proximities, (int)):
            raise ValueError(
                f"`max_proximities` must be an integer if not None, got {max_proximities}."
            )

        if oob_score:
            if not self.bootstrap:
                raise ValueError("Out-of-bag estimation only available if bootstrap=True.")

            X_leaves, X_indices = self._oob_samples(X, indices, duplicates)

            if (X_indices.sum(axis=1) == 0).any():
                warn(
                    "Some inputs do not have OOB scores. "
                    "This probably means too few trees were used "
                    "to compute any reliable OOB estimates."
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                X_leaves = self.apply(X)
            X_indices = None

        proximities = self.forest_.proximity_counts(
            X_leaves,
            X_indices,
            max_proximities,
            self.sorter_,
        )

        if return_sorted:
            # Sort each dict of proximities in descending order by count.
            proximities = [
                sorted(p.items(), key=lambda x: x[1], reverse=True) for p in proximities
            ]
        else:
            proximities = [p.items() for p in proximities]

        return proximities

    def score(self, X, y, quantiles=0.5, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        quantiles : float or list, default=0.5
            The quantile or list of quantiles that the model tries to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `y`.
        """
        from sklearn.metrics import r2_score

        return r2_score(
            y,
            self.predict(X, quantiles),
            sample_weight=sample_weight,
            multioutput="variance_weighted",
        )


class RandomForestQuantileRegressor(BaseForestQuantileRegressor):
    """A random forest regressor that provides quantile estimates.

    A quantile random forest is a meta estimator that fits a number of
    decision trees on various sub-samples of the dataset, keeps the values of
    samples that reach each node, and assesses the conditional distribution
    based on this information [1]. The sub-sample size is controlled with the
    `max_samples` parameter if `bootstrap=True` (default), otherwise the whole
    dataset is used to build each tree. The leaf-sample size is controlled
    with the `max_samples_leaf` parameter.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    default_quantiles : float, list, or "mean", default=0.5
        The default quantile or list of quantiles that the model tries to
        predict. Each quantile must be strictly between 0 and 1. If "mean",
        the model predicts the mean.

        .. versionadded:: 1.2

    criterion : {"squared_error", "absolute_error", "friedman_mse", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.
        Training using "absolute_error" is significantly slower
        than when using "squared_error".

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    max_samples_leaf : int, float or None, default=1
        The maximum number of samples permitted to be at a leaf node.

        - If int, then consider `max_samples_leaf` as the maximum number.
        - If float, then `max_samples_leaf` is a fraction and
          `ceil(max_samples_leaf * n_samples)` are the maximum
          number of samples for each node.
        - If None, then unlimited number of leaf samples.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default=1.0
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered
          at each split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None or 1.0, then `max_features=n_features`.

        .. note::
            The default of 1.0 is equivalent to bagged trees and more
            randomness can be achieved by setting smaller values, e.g. 0.3.

        .. sklearn-versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to 1.0.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None, then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool or callable, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        By default, :func:`~sklearn.metrics.r2_score` is used.
        Provide a callable with signature `metric(y_true, y_pred)` to use a
        custom metric. Only available if `bootstrap=True`.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max(round(n_samples * max_samples), 1)` samples.
          Thus, `max_samples` should be in the interval `(0.0, 1.0]`.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonically increasing
          - 0: no constraint
          - -1: monotonically decreasing

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multioutput regressions (i.e. when `n_outputs_ > 1`),
          - regressions trained on data with missing values,
          - trees with multi-sample leaves (i.e. when `max_samples_leaf > 1`).

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.DecisionTreeRegressor`
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    See Also
    --------
    ExtraTreesQuantileRegressor : Quantile ensemble of extremely randomized
        tree regressors.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    The default value ``max_features=1.0`` uses ``n_features``
    rather than ``n_features / 3``. The latter was originally suggested in
    [2], whereas the former was more recently justified empirically in [3].

    References
    ----------
    .. [1] N. Meinshausen, "Quantile Regression Forests", Journal of Machine
           Learning Research, 7(Jun), 983-999, 2006.
           http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf

    .. [2] \\L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    .. [3] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely Randomized
           Trees", Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=10000, n_features=8, random_state=0)
    >>> qrf = RandomForestQuantileRegressor(random_state=0)
    >>> qrf.fit(X[:1000], y[:1000])
    RandomForestQuantileRegressor(random_state=0)
    >>> qrf.score(X, y, quantiles=0.5)
    0.859...
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        default_quantiles=0.5,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        """Initialize random forest quantile regressor."""
        init_dict = {
            "estimator": DecisionTreeRegressor(),
            "n_estimators": n_estimators,
            "estimator_params": (
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            "bootstrap": bootstrap,
            "oob_score": oob_score,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "warm_start": warm_start,
            "max_samples": max_samples,
            "max_samples_leaf": max_samples_leaf,
        }
        super(RandomForestQuantileRegressor, self).__init__(**init_dict)

        self.default_quantiles = default_quantiles
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_samples_leaf = max_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst


class ExtraTreesQuantileRegressor(BaseForestQuantileRegressor):
    """An extra-trees regressor that provides quantile estimates.

    A quantile extra trees regressor is a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples of
    the dataset, keeps the values of samples that reach each node, and
    assesses the conditional distribution based on this information [1]. The
    leaf-sample size is controlled with the `max_samples_leaf` parameter.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    default_quantiles : float, list, or "mean", default=0.5
        The default quantile or list of quantiles that the model tries to
        predict. Each quantile must be strictly between 0 and 1. If "mean",
        the model predicts the mean.

        .. versionadded:: 1.2

    criterion : {"squared_error", "absolute_error", "friedman_mse", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.
        Training using "absolute_error" is significantly slower
        than when using "squared_error".

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    max_samples_leaf : int, float or None, default=1
        The maximum number of samples permitted to be at a leaf node.

        - If int, then consider `max_samples_leaf` as the maximum number.
        - If float, then `max_samples_leaf` is a fraction and
          `ceil(max_samples_leaf * n_samples)` are the maximum
          number of samples for each node.
        - If None, then unlimited number of leaf samples.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default=1.0
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered
          at each split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None or 1.0, then `max_features=n_features`.

        .. note::
            The default of 1.0 is equivalent to bagged trees and more
            randomness can be achieved by setting smaller values, e.g. 0.3.

        .. sklearn-versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to 1.0.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None, then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool or callable, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        By default, :func:`~sklearn.metrics.r2_score` is used.
        Provide a callable with signature `metric(y_true, y_pred)` to use a
        custom metric. Only available if `bootstrap=True`.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls 3 sources of randomness:

        - the bootstrapping of the samples used when building trees
          (if ``bootstrap=True``)
        - the sampling of the features to consider when looking for the best
          split at each node (if ``max_features < n_features``)
        - the draw of the splits for each of the `max_features`

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonically increasing
          - 0: no constraint
          - -1: monotonically decreasing

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multioutput regressions (i.e. when `n_outputs_ > 1`),
          - regressions trained on data with missing values,
          - trees with multi-sample leaves (i.e. when `max_samples_leaf > 1`).

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.ExtraTreeRegressor`
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature. It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    See Also
    --------
    RandomForestQuantileRegressor : Quantile ensemble regressor using trees.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------
    .. [1] N. Meinshausen, "Quantile Regression Forests", Journal of Machine
           Learning Research, 7(Jun), 983-999, 2006.
           http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf

    .. [2] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely Randomized Trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from quantile_forest import ExtraTreesQuantileRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=10000, n_features=8, random_state=0)
    >>> qrf = ExtraTreesQuantileRegressor(random_state=0)
    >>> qrf.fit(X[:1000], y[:1000])
    ExtraTreesQuantileRegressor(random_state=0)
    >>> qrf.score(X, y, quantiles=0.5)
    0.894...
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        default_quantiles=0.5,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        """Initialize extra trees quantile regressor."""
        init_dict = {
            "estimator": ExtraTreeRegressor(),
            "n_estimators": n_estimators,
            "estimator_params": (
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            "bootstrap": bootstrap,
            "oob_score": oob_score,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "warm_start": warm_start,
            "max_samples": max_samples,
            "max_samples_leaf": max_samples_leaf,
        }
        super(ExtraTreesQuantileRegressor, self).__init__(**init_dict)

        self.default_quantiles = default_quantiles
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_samples_leaf = max_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
