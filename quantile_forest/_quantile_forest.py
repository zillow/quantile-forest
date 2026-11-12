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

import hashlib
import random
from abc import abstractmethod
from math import ceil
from numbers import Integral, Real
from warnings import warn

import joblib
import numpy as np
from scipy.sparse import issparse
from sklearn.ensemble._forest import (
    ForestRegressor,
    _generate_sample_indices,
    _get_n_samples_bootstrap,
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    validate_data,
)

from ._quantile_forest_fast import QuantileForest
from ._utils import group_indices_by_value


def _check_quantiles(quantiles):
    """Validate quantiles and return them as a list of floats.

    Returns ``[-1]`` if the mean is requested instead.
    """
    if isinstance(quantiles, str) and quantiles == "mean":
        return [-1]
    if isinstance(quantiles, list) and quantiles == ["mean"]:
        return [-1]

    try:
        quantiles_array = np.asarray(quantiles)
    except ValueError:  # e.g., ragged nested lists
        quantiles_array = None
    if (
        quantiles_array is None
        or quantiles_array.dtype.kind not in "biuf"  # boolean, integer, or float
        or quantiles_array.ndim > 1
    ):
        raise ValueError(
            f'Quantiles must be a float, an array-like of floats, or "mean", got {quantiles!r}.'
        )

    quantiles = quantiles_array.astype(np.float64).reshape(-1).tolist()
    if not quantiles:
        raise ValueError("Quantiles must not be empty.")
    for q in quantiles:
        if not 0 <= q <= 1:  # also rejects NaN
            raise ValueError(f"Quantiles must be in the range [0, 1], got {q}.")
    return quantiles


def _check_criterion(criterion):
    """Return the criterion, with the deprecated "friedman_mse" replaced.

    Scikit-learn 1.9 deprecated "friedman_mse", which was always equivalent
    to "squared_error" for forests. Its forests replace it (with a warning)
    when they are initialized, so do the same.
    """
    # TODO: Remove when scikit-learn removes "friedman_mse" (in version 1.11).
    if isinstance(criterion, str) and criterion == "friedman_mse":
        warn(
            'Value `"friedman_mse"` for `criterion` is deprecated and will be '
            "removed when scikit-learn removes it (in version 1.11). It maps to "
            '`"squared_error"` as both were always equivalent. Use '
            '`criterion="squared_error"` to remove this warning.',
            FutureWarning,
            stacklevel=3,
        )
        return "squared_error"
    return criterion


# The dtype of the samples that scikit-learn's trees are fit on.
DTYPE = np.float32

# The training indices of the leaf nodes are stored as 32-bit integers.
_MAX_SAMPLES = np.iinfo(np.int32).max

# Dense input samples are hashed in blocks of rows of about this many bytes.
_HASH_BLOCK_BYTES = 2**22


def _hash_X(X):
    """Hash input samples so that equal data gives equal hashes.

    Fit validates sparse inputs as CSC and predict validates them as CSR, so
    sparse inputs are hashed in canonical CSC form, independent of the format
    and container they were passed in.

    Dense inputs are hashed row by row, independent of their memory layout
    (e.g., the values of a pandas DataFrame are stored column by column, and
    those of a NumPy array usually row by row), a block of rows at a time so
    that the input is not copied.
    """
    if issparse(X):
        X = X.tocsc()  # no copy if already CSC, as it is during fit
        if not X.has_canonical_format:
            X = X.copy()
            X.sum_duplicates()  # also sorts the indices
        return joblib.hash((X.shape, X.data, X.indices, X.indptr))

    X = np.asarray(X)
    hasher = hashlib.md5(usedforsecurity=False)
    hasher.update(repr((X.shape, X.dtype.str)).encode())
    n_rows = max(1, _HASH_BLOCK_BYTES // max(1, X[:1].nbytes))
    for start in range(0, X.shape[0], n_rows):
        # No copy if the rows are already contiguous.
        hasher.update(np.ascontiguousarray(X[start : start + n_rows]))
    return hasher.hexdigest()


def _matches_hash_X(X, X_hash):
    """Return whether input samples match the hash of other input samples.

    Models fit by version 1.5.0 or earlier hashed dense inputs with joblib,
    in their memory layout, so their hashes are matched in that way too.
    """
    if _hash_X(X) == X_hash:
        return True
    return not issparse(X) and joblib.hash(X) == X_hash


def _nanquantile_rows(a, quantiles, method):
    """Return the quantiles of each row of `a`, ignoring NaNs.

    Equivalent to ``np.nanquantile(a, quantiles, method=method, axis=1)``,
    which loops over the rows in Python. Instead, sort the rows, which puts
    NaNs last, and compute the quantiles of all rows with the same number of
    values at once. Rows of only NaNs give NaNs.
    """
    n_values = np.count_nonzero(~np.isnan(a), axis=1)
    a = np.sort(a, axis=1)
    out = np.full((len(quantiles), a.shape[0]), np.nan)
    for n in np.unique(n_values[n_values > 0]):
        rows = n_values == n
        out[:, rows] = np.quantile(a[rows, :n], quantiles, method=method, axis=1)
    return out


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

    def fit(self, X, y, sample_weight=None):
        """Build a forest from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``. It can contain missing
            values (NaN) wherever a scikit-learn forest accepts them. See
            :ref:`user-guide-fit-predict`.

            .. versionadded:: 1.5
                Support for missing values.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. The leaf samples
            used for prediction are weighted too, so that a sample with weight
            2 counts like two samples with weight 1. (With bootstrapping, the
            bootstrap samples are drawn in proportion to the weights instead.)
            Only the relative weights matter: where the number of samples does
            too (e.g., for quantiles that interpolate between samples), a
            sample of average positive weight counts as one sample. Samples
            with zero or negative weight are excluded from the leaf samples.
            See :ref:`user-guide-fit-predict`.

            .. versionchanged:: 1.5
                The leaf samples are weighted as described above. Earlier
                versions only left out the leaf samples with zero weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        if self.default_quantiles is not None:
            _check_quantiles(self.default_quantiles)

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

        X, y = validate_data(
            self,
            X=X,
            y=y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            # Any missing values that are not supported have been rejected.
            ensure_all_finite=False,
        )

        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        # Sort the target values in ascending order.
        # Use sorter to maintain mapping to original order.
        sorter = np.argsort(y, axis=0)
        y_sorted = np.empty_like(y)
        for i in range(y.shape[1]):
            y_sorted[:, i] = y[sorter[:, i], i]

        if sample_weight is not None:
            # Validate the weights as the forest did when it drew its bootstrap
            # samples, so that the stored weights redraw the same samples.
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight_sorted = sample_weight[sorter]
        else:
            sample_weight_sorted = None

        self.sample_weight_ = sample_weight

        # Get the weights of the leaf samples, if they are weighted. With an
        # integer `max_samples_leaf=1`, each leaf keeps one sample, so pick it
        # in proportion to its weight instead.
        leaf_weight = self._get_leaf_weight(sample_weight)
        weighted_subsample = leaf_weight is not None and (
            isinstance(self.max_samples_leaf, (Integral, np.integer))
            and self.max_samples_leaf == 1
        )
        if weighted_subsample:
            leaf_weight = None

        # Get the training indices of the leaf nodes of each tree.
        tree_offsets, leaf_offsets, y_train_indices = self._get_y_train_leaves(
            X,
            y_sorted,
            sorter=sorter,
            sample_weight=sample_weight_sorted,
            weighted_subsample=weighted_subsample,
        )

        # Get map of tree leaf nodes to target value bounds (for monotonicity constraints).
        y_bound_leaves = self._get_y_bound_leaves(
            y_sorted, tree_offsets, leaf_offsets, y_train_indices
        )

        # Create quantile forest object.
        self.forest_ = QuantileForest(
            y_sorted.astype(np.float64).T,
            y_bound_leaves=y_bound_leaves,
            sample_weight=None if leaf_weight is None else leaf_weight[sorter].T,
            tree_offsets=tree_offsets,
            leaf_offsets=leaf_offsets,
            y_train_indices=y_train_indices,
        )

        self.sorter_ = sorter
        self.n_train_samples_ = len(y_sorted)
        self.X_train_hash_ = _hash_X(X)
        self.unsampled_indices_ = None

        # Record the samples left out of each tree, for out-of-bag estimation.
        self._unsampled_masks = None
        if self.bootstrap:
            self._unsampled_masks = self._get_unsampled_masks()

        return self

    def _check_X_for_trees(self, X):
        """Check input samples as ``apply`` does, except for their feature names.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples, which have already been validated against the
            fitted features (e.g., by ``_validate_X_predict``), so that they
            have no feature names left to check.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples as the trees expect them, with
            ``dtype=np.float32`` and sparse matrices as ``csr_matrix``.
        """
        # Missing values are only valid where the trees support them.
        allow_nan = self.estimators_[0]._support_missing_values(X)
        X = check_array(
            X,
            accept_sparse="csr",
            dtype=DTYPE,
            ensure_all_finite="allow-nan" if allow_nan else True,
            estimator=self,
            input_name="X",
        )
        if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
            raise ValueError("No support for np.int64 index based sparse matrices")
        return X

    def _apply_validated(self, X):
        """Apply the trees to X and return the leaf indices, as ``apply`` does.

        Unlike ``apply``, this does not validate X against the fitted features.
        The callers have already validated X, so that it has no feature names,
        and ``apply`` would warn that it has none. Ignoring that warning would
        change the warning filters of the whole process, which is not
        thread-safe: concurrent calls could leave the warnings ignored.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples, which have already been validated.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each sample and each tree, the index of the leaf that the
            sample ends up in.
        """
        check_is_fitted(self)
        X = self._check_X_for_trees(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads")(
            delayed(tree.apply)(X, check_input=False) for tree in self.estimators_
        )
        return np.array(results).T

    def _get_leaf_weight(self, sample_weight):
        """Return the weights of the training samples in the leaf nodes.

        Scikit-learn fits each tree with the sample weights, except that with
        bootstrapping, it draws the bootstrap samples in proportion to the
        weights and fits each tree with the counts of the samples drawn. So
        the leaf samples are weighted only if the trees are fit with the
        weights, so that a sample with weight 2 counts like two samples with
        weight 1.

        Parameters
        ----------
        sample_weight : array-like of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        leaf_weight : array-like of shape (n_samples,)
            Weights of the training samples in the leaf nodes, which are 0 for
            samples with zero or negative weight. None if the leaf samples are
            not weighted, including if all positive weights are equal.
        """
        if sample_weight is None:
            return None

        if self.bootstrap:
            return None  # the bootstrap samples are drawn in proportion to the weights

        positive = sample_weight > 0
        positive_weight = sample_weight[positive]
        if not np.any(positive_weight != positive_weight[:1]):
            return None  # samples with zero weight are already excluded

        # Keep the weights as given (e.g., exact integers), since predictions
        # scale them where the number of samples matters.
        return np.where(positive, sample_weight, 0).astype(np.float64)

    @staticmethod
    def _get_y_train_leaves_slice(
        bootstrap_indices,
        X_leaves_bootstrap,
        sample_weight,
        leaf_subsample,
        max_node_count,
        max_samples_leaf,
        random_state,
        weighted_subsample=False,
    ):
        """Return the training sample indices of a tree's leaf nodes.

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
            Maximum number of nodes (internal + leaves) across all trees.

        max_samples_leaf : int
            Maximum number of samples per leaf node.

        random_state : int, RandomState instance
            Controls the sampling of the training indices at each leaf node.

        weighted_subsample : bool, default=False
            Sample the training indices at each leaf node in proportion to
            their weights. Only used if `max_samples_leaf=1`.

        Returns
        -------
        leaf_counts : array-like of shape (max_node_count,)
            Number of training samples of each node. Nodes with no samples
            (e.g., internal nodes) are included, so that node indices match
            their ``est.apply`` outputs.

        y_train_indices : array-like of shape (n_outputs, n_stored)
            One-based indices of the training samples of each node, ordered
            by node.
        """
        # Group training indices by leaf node.
        leaf_indices, leaf_values_list = group_indices_by_value(X_leaves_bootstrap)

        weights = None
        positive_mask = None
        if sample_weight is not None:
            # Weights are shared across outputs, so output 0 suffices.
            weights = sample_weight[bootstrap_indices[:, 0] - 1, 0]
            positive_mask = weights > 0
            if positive_mask.all():
                positive_mask = None
        weighted_subsample = weighted_subsample and weights is not None and max_samples_leaf == 1

        if leaf_subsample or positive_mask is not None:
            if leaf_subsample:
                rng = random.Random(random_state)

            for j in range(len(leaf_values_list)):
                if positive_mask is not None:
                    leaf_values_list[j] = leaf_values_list[j][positive_mask[leaf_values_list[j]]]
                if leaf_subsample:
                    # Sample leaf to length `max_samples_leaf`.
                    if len(leaf_values_list[j]) > max_samples_leaf:
                        if weighted_subsample:
                            # Pick one sample with probability proportional to its weight.
                            leaf_weights = weights[leaf_values_list[j]]
                            leaf_values_list[j] = rng.choices(leaf_values_list[j], leaf_weights)
                        else:
                            rng.shuffle(leaf_values_list[j])  # to ensure random sampling
                            leaf_values_list[j] = leaf_values_list[j][:max_samples_leaf]

        # Count the training indices of each node, and list them in node order.
        # Leaves left without any (e.g., if all of their samples have zero
        # weight) stay empty.
        leaf_counts = np.zeros(max_node_count, dtype=np.int64)
        leaf_counts[leaf_indices] = [len(values) for values in leaf_values_list]
        leaf_values = np.concatenate(leaf_values_list)
        y_train_indices = np.ascontiguousarray(bootstrap_indices[leaf_values].T, dtype=np.int32)

        return leaf_counts, y_train_indices

    def _get_y_train_leaves(self, X, y, sorter=None, sample_weight=None, weighted_subsample=False):
        """Return the training indices of the leaf nodes of each tree.

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

        weighted_subsample : bool, default=False
            Sample the training indices at each leaf node in proportion to
            their weights. Only used if `max_samples_leaf=1`.

        Returns
        -------
        tree_offsets : array-like of shape (n_estimators + 1,)
            Position in `y_train_indices` of the first training sample of
            each tree, followed by the number of training samples stored.

        leaf_offsets : array-like of shape (n_estimators, n_nodes + 1)
            Position of the first training sample of each node, relative to
            the position of its tree, followed by the number of training
            samples of the tree. Nodes with no samples (e.g., internal nodes)
            are empty. Internal nodes are included so that node indices match
            their ``est.apply`` outputs.

        y_train_indices : array-like of shape (n_outputs, n_stored)
            One-based indices of the training samples of each node, ordered
            by tree and node. Those of node ``n`` of tree ``t`` are at
            positions ``tree_offsets[t] + leaf_offsets[t, n]`` up to (but
            excluding) ``tree_offsets[t] + leaf_offsets[t, n + 1]``.
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

        X_leaves = self._apply_validated(X)

        if self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples, self.max_samples, self.sample_weight_
            )

        n_estimators = len(self.estimators_)
        shape = (n_samples if not self.bootstrap else n_samples_bootstrap, n_estimators)

        # The training indices, and the offsets within each tree (which count
        # its bootstrap samples), are stored as 32-bit integers.
        if max(shape[0], n_samples) > _MAX_SAMPLES:
            raise ValueError(
                f"At most {_MAX_SAMPLES} training samples, and bootstrap samples per tree, "
                f"are supported, got {n_samples} and {shape[0]}."
            )

        bootstrap_indices = np.empty(shape, dtype=np.int64)
        X_leaves_bootstrap = np.empty(shape, dtype=np.int64)
        for i, estimator in enumerate(self.estimators_):
            # Get bootstrap indices.
            if self.bootstrap:
                bootstrap_indices[:, i] = _generate_sample_indices(
                    estimator.random_state, n_samples, n_samples_bootstrap, self.sample_weight_
                )
            else:
                bootstrap_indices[:, i] = np.arange(n_samples)

            # Get leaf node indices of bootstrap training samples.
            X_leaves_bootstrap[:, i] = X_leaves[bootstrap_indices[:, i], i]

        if sorter is not None:
            # Reassign bootstrap indices to account for target sorting.
            bootstrap_indices = np.argsort(sorter, axis=0)[bootstrap_indices]

        bootstrap_indices = bootstrap_indices.reshape(-1, n_estimators, n_outputs)
        bootstrap_indices += 1  # one-based

        # Get the maximum number of nodes (internal + leaves) across trees.
        max_node_count = max(estimator.tree_.node_count for estimator in self.estimators_)

        leaf_offsets = np.zeros((n_estimators, max_node_count + 1), dtype=np.int32)
        y_train_indices = []
        for i, estimator in enumerate(self.estimators_):
            leaf_counts, tree_indices = self._get_y_train_leaves_slice(
                bootstrap_indices[:, i],
                X_leaves_bootstrap[:, i],
                sample_weight,
                leaf_subsample,
                max_node_count,
                max_samples_leaf,
                estimator.random_state,
                weighted_subsample=weighted_subsample,
            )
            np.cumsum(leaf_counts, out=leaf_offsets[i, 1:])
            y_train_indices.append(tree_indices)

        tree_offsets = np.zeros(n_estimators + 1, dtype=np.int64)
        np.cumsum(leaf_offsets[:, -1], out=tree_offsets[1:])

        return tree_offsets, leaf_offsets, np.concatenate(y_train_indices, axis=1)

    def _get_y_bound_leaves(self, y, tree_offsets, leaf_offsets, y_train_indices):
        """Return the bounds for target values for each leaf node.

        The target value bounds are used to enforce monotonicity constraints.

        Parameters
        ----------
        y : array-like of shape (n_samples, n_outputs)
            The target outputs for each y value.

        tree_offsets, leaf_offsets, y_train_indices : array-like
            The training indices of the leaf nodes of each tree, as returned
            by `_get_y_train_leaves`.

        Returns
        -------
        y_bound_leaves : array-like of shape (n_estimators, n_leaves, 2)
            Minimum and maximum bounds for target values for each leaf node.
            Used to enforce monotonicity constraints. None if no constraints.
        """
        if self.monotonic_cst is None:
            return None

        monotonic_cst = np.asarray(self.monotonic_cst)
        n_nodes = leaf_offsets.shape[1] - 1
        y_bound_leaves = np.full((len(self.estimators_), n_nodes, 2), [-np.inf, np.inf])

        # Process each tree one depth level at a time, rather than node by node.
        for i, estimator in enumerate(self.estimators_):
            tree = estimator.tree_
            children_left = tree.children_left
            children_right = tree.children_right
            is_leaf = children_left == children_right

            # Group the nodes by depth, from the root down.
            levels = [np.array([0])]
            while True:
                parents = levels[-1][~is_leaf[levels[-1]]]
                if parents.size == 0:
                    break
                levels.append(np.concatenate([children_left[parents], children_right[parents]]))

            # Count and sum the stored targets of the leaves (of the only output,
            # as constraints do not support multiple outputs), then of the
            # non-leaf nodes from their children, from the deepest level up.
            leaf_counts = np.diff(leaf_offsets[i, : tree.node_count + 1])
            stored_nodes = np.repeat(np.arange(tree.node_count), leaf_counts)
            stored_values = y[y_train_indices[0, tree_offsets[i] : tree_offsets[i + 1]] - 1, 0]
            num_values = leaf_counts.astype(np.float64)
            sum_values = np.bincount(stored_nodes, stored_values, minlength=tree.node_count)
            for nodes in reversed(levels):
                nodes = nodes[~is_leaf[nodes]]
                left, right = children_left[nodes], children_right[nodes]
                num_values[nodes] = num_values[left] + num_values[right]
                sum_values[nodes] = sum_values[left] + sum_values[right]

            # Pass bounds from the root down to enforce monotonicity: a split on
            # a constrained feature bounds its children at the midpoint of their
            # mean targets (clipped to the node's own bounds).
            constraints = np.zeros(tree.node_count, dtype=int)
            constraints[~is_leaf] = monotonic_cst[tree.feature[~is_leaf]]
            with np.errstate(divide="ignore", invalid="ignore"):
                half_means = sum_values / (2 * num_values)
            lower = np.full(tree.node_count, -np.inf)
            upper = np.full(tree.node_count, np.inf)
            for nodes in levels:
                nodes = nodes[~is_leaf[nodes]]
                left, right = children_left[nodes], children_right[nodes]
                min_bound, max_bound = lower[nodes], upper[nodes]
                mid = half_means[left] + half_means[right]
                mid = np.maximum(np.minimum(mid, max_bound), min_bound)
                cst = constraints[nodes]
                lower[left] = np.where(cst == -1, mid, min_bound)  # decreasing
                upper[left] = np.where(cst == 1, mid, max_bound)  # increasing
                lower[right] = np.where(cst == 1, mid, min_bound)
                upper[right] = np.where(cst == -1, mid, max_bound)

            leaves = np.flatnonzero(is_leaf)
            y_bound_leaves[i, leaves, 0] = lower[leaves]
            y_bound_leaves[i, leaves, 1] = upper[leaves]

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
            elif not _matches_hash_X(X, self.X_train_hash_):
                warn("OOB samples are not identical to training samples.")

        if indices is not None and n_samples != len(indices):
            raise ValueError("If `indices` are not None, X and indices must be the same length.")

        # Get the training index of each sample, unless it is omitted (-1).
        if indices is None:
            train_indices = np.arange(n_samples)
            has_index = np.ones(n_samples, dtype=bool)
        else:
            indices = np.asarray(indices)
            if indices.ndim != 1 or indices.dtype.kind not in "iuf":  # integers or floats
                raise ValueError(
                    "`indices` must be a one-dimensional array-like of integers, "
                    f"got an array of shape {indices.shape} and dtype {indices.dtype}."
                )
            with np.errstate(invalid="ignore"):
                train_indices = indices.astype(np.intp)
            has_index = (
                (train_indices == indices)
                & (train_indices >= 0)
                & (train_indices < self.n_train_samples_)
            )
            is_invalid = ~has_index & (indices != -1)
            if is_invalid.any():
                raise ValueError(
                    "`indices` must be training indices, from 0 up to (but excluding) "
                    f"{self.n_train_samples_}, or -1 for samples omitted from the training "
                    f"set, got {indices[is_invalid][0]}."
                )

        # Look up the bits of the training indices excluded from each tree's
        # bootstrap sample.
        unsampled_masks = self._get_unsampled_masks(duplicates)
        train_indices = train_indices[has_index]
        shifts = (7 - (train_indices & 7)).astype(np.uint8)
        unsampled_bits = (unsampled_masks[:, train_indices >> 3] >> shifts) & 1

        X_indices = np.zeros((n_samples, n_estimators), dtype=np.uint8)
        X_indices[has_index] = unsampled_bits.T
        if indices is not None:
            # Token index (-1) denotes indices that are always unsampled.
            X_indices[indices == -1] = 1

        # Check data once for all trees, as `apply` does. Callers pass arrays
        # they have already validated, which have no feature names to check.
        X = self._check_X_for_trees(X)

        X_leaves = np.empty((n_samples, n_estimators), dtype=np.intp)
        for i, estimator in enumerate(self.estimators_):
            unsampled_indices = np.flatnonzero(X_indices[:, i])
            if unsampled_indices.size > 0:
                # Get leaf node indices for unsampled training samples.
                X_leaves[unsampled_indices, i] = estimator.apply(
                    X[unsampled_indices], check_input=False
                )

        return X_leaves, X_indices

    def _get_unsampled_masks(self, duplicates=None):
        """Get the unsampled indices of all base estimators as bit masks.

        Parameters
        ----------
        duplicates : list of lists, default=None
            List of sets of functionally identical indices.

        Returns
        -------
        unsampled_masks : array of shape (n_estimators, ceil(n_train_samples / 8))
            For each base estimator, a mask of the training indices excluded
            from its bootstrap sample, with bits packed by ``np.packbits``.
        """
        n_estimators = len(self.estimators_)
        n_train_samples = self.n_train_samples_

        if self.unsampled_indices_ is not None:  # precomputed unsampled indices
            masks = np.zeros((n_estimators, n_train_samples), dtype=bool)
            for i, unsampled_indices in enumerate(self.unsampled_indices_):
                unsampled_indices = np.asarray(unsampled_indices, dtype=np.intp)
                in_range = (unsampled_indices >= 0) & (unsampled_indices < n_train_samples)
                masks[i, unsampled_indices[in_range]] = True
            return np.packbits(masks, axis=1)

        if not duplicates and getattr(self, "_unsampled_masks", None) is not None:
            return self._unsampled_masks  # stored during fit

        if not self.bootstrap:
            warn("Unsampled indices only exist if bootstrap=True.")
            return np.zeros((n_estimators, (n_train_samples + 7) // 8), dtype=np.uint8)

        groups = []
        for group in duplicates or []:
            group = np.asarray(list(group), dtype=np.intp)
            groups.append(group[(group >= 0) & (group < n_train_samples)])

        masks = np.zeros((n_estimators, (n_train_samples + 7) // 8), dtype=np.uint8)
        for i, estimator in enumerate(self.estimators_):
            masks[i] = np.packbits(self._get_unsampled_mask(estimator, groups))
        return masks

    def _get_unsampled_mask(self, estimator, duplicates=None):
        """Get a boolean mask of the unsampled indices for a base estimator.

        Parameters
        ----------
        estimator : object
            Base estimator.

        duplicates : list of array-likes, default=None
            List of sets of functionally identical indices.

        Returns
        -------
        unsampled_mask : array of shape (n_train_samples,)
            True for the training indices excluded from the bootstrap sample.
        """
        # Redraw the bootstrap sample as scikit-learn does (e.g., for its
        # `estimators_samples_`), from what the forest stored when it was fit,
        # possibly with an earlier version of scikit-learn (e.g., in a model
        # pickled by an earlier version): before 1.9, the samples were drawn
        # without the sample weights, which 1.9 and later store, and before 1.4,
        # the number of samples drawn was not stored, so it is recomputed.
        sample_weight = getattr(self, "_sample_weight", None)
        n_train_samples = self.n_train_samples_
        n_samples_bootstrap = getattr(self, "_n_samples_bootstrap", None)
        if n_samples_bootstrap is None:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_train_samples, self.max_samples, sample_weight
            )
        sample_indices = _generate_sample_indices(
            estimator.random_state, n_train_samples, n_samples_bootstrap, sample_weight
        )
        sampled_mask = np.bincount(sample_indices, minlength=n_train_samples) > 0

        # Treat a set of duplicates as sampled if any of them is sampled.
        for group in duplicates or []:
            group = np.asarray(list(group), dtype=np.intp)
            if sampled_mask[group].any():
                sampled_mask[group] = True

        return ~sampled_mask

    def _clipped_quantile_ranks(self, y, X_leaves, X_indices, kind, aggregate_leaves_first):
        """Return quantile ranks among the clipped leaf values.

        With monotonicity constraints, each tree keeps one value per leaf,
        which predict clips to the bounds of the leaf. Rank the scores among
        the clipped values, as the quantile ranks kernel ranks them among the
        values that the leaves keep.

        Parameters
        ----------
        y : array-like of shape (n_samples, n_outputs)
            The target values for which to calculate ranks.

        X_leaves : array-like of shape (n_samples, n_estimators)
            Target leaf node indices along samples and trees.

        X_indices : array-like of shape (n_samples, n_estimators)
            Mask for the trees to use for each sample. 1 if used, 0 otherwise.
            If None, all trees are used for all samples.

        kind : {b"rank", b"weak", b"strict", b"mean"}
            Specifies the interpretation of the resulting score.

        aggregate_leaves_first : bool
            Calculate ranks using leaf values aggregated across trees.

        Returns
        -------
        y_ranks : array of shape (n_samples, n_outputs)
            Quantile ranks in range [0, 1]. NaN if a sample has no trees.
        """
        y_bound_leaves = np.asarray(self.forest_.y_bound_leaves)
        y_train = np.asarray(self.forest_.y_train).T
        n_samples, n_trees = X_leaves.shape
        n_outputs = y_train.shape[1]

        if y.shape[1] != n_outputs:
            raise ValueError(
                f"Number of target outputs in training data ({n_outputs}) does not "
                f"match the number of outputs in test data being scored ({y.shape[1]})."
            )

        kinds = [b"rank", b"weak", b"strict", b"mean"]
        if kind not in kinds:
            raise ValueError(
                f"Invalid kind {bytes(kind).decode(errors='replace')!r}, which must be one of: "
                f"{b', '.join(kinds).decode()}."
            )

        y_ranks = np.full((n_samples, n_outputs), np.nan)
        for output in range(n_outputs):
            # Get each tree's clipped leaf value, or NaN if the tree is unused.
            leaf_vals = np.full((n_samples, n_trees), np.nan)
            for tree in range(n_trees):
                rows = slice(None) if X_indices is None else X_indices[:, tree] == 1
                leaves = X_leaves[rows, tree]
                train_indices = self.forest_.first_train_indices(tree, leaves, output)
                leaf_vals[rows, tree] = np.where(
                    train_indices > 0,  # empty leaves have NaN values
                    np.clip(
                        y_train[train_indices - 1, output],
                        y_bound_leaves[tree, leaves, 0],
                        y_bound_leaves[tree, leaves, 1],
                    ),
                    np.nan,
                )

            # Count the values below (left) and up to (right) each score within
            # each tree, or across all trees if aggregating leaves first.
            scores = y[:, [output]]
            n_values = np.count_nonzero(~np.isnan(leaf_vals), axis=1)
            left = (leaf_vals < scores).astype(np.intp)  # NaN values are not counted
            right = (leaf_vals <= scores).astype(np.intp)
            if aggregate_leaves_first:
                left, right = left.sum(axis=1), right.sum(axis=1)

            if kind == b"rank":
                ranks = (left + right + (right > left)) * 0.5
            elif kind == b"weak":
                ranks = right
            elif kind == b"strict":
                ranks = left
            else:  # mean
                ranks = (left + right) * 0.5

            if not aggregate_leaves_first:
                ranks = ranks.sum(axis=1)  # to average the ranks of the trees

            has_values = n_values > 0
            y_ranks[has_values, output] = ranks[has_values] / n_values[has_values]

        return y_ranks

    def predict(
        self,
        X,
        quantiles=None,
        interpolation="averaged_inverted_cdf",
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

        quantiles : float, array-like, or "mean", default=None
            The quantile or list of quantiles that the model tries to predict.
            Each quantile must be in the range [0, 1]. If "mean", the model
            predicts the mean. If None, the model uses the value of
            `default_quantiles`.

        interpolation : {"averaged_inverted_cdf", "inverted_cdf", "linear", \
                "lower", "higher", "midpoint", "nearest"}, \
                default="averaged_inverted_cdf"
            Specifies how to calculate a quantile from the training samples:

            - If "averaged_inverted_cdf", then the smallest training target
              such that a fraction (at least) ``q`` of the weighted training
              samples are at or below it, except that if the fraction is
              exactly ``q``, that target and the next one are averaged, so
              that the median is the usual median.
            - If "inverted_cdf", then the same, without averaging.

            These match NumPy's methods of the same names, and the default
            matches scikit-learn's weighted quantiles (e.g., the leaf values
            of trees fit with ``criterion="absolute_error"``). They depend
            only on the relative weights of the training samples, so a sample
            with weight 2 counts exactly like two samples with weight 1.

            The other methods interpolate when the desired quantile lies
            between two data points ``i < j``, as NumPy's methods of the same
            names do:

            - If "linear", then ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i`` and
              ``j``.
            - If "lower", then ``i``.
            - If "higher", then ``j``.
            - If "nearest", then ``i`` or ``j``, whichever is nearest.
            - If "midpoint", then ``(i + j) / 2``.

            .. versionchanged:: 1.5
                The default changed from "linear" to "averaged_inverted_cdf".

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
            If the leaf samples are weighted by their sample weights (see the
            `sample_weight` parameter of `fit`), weighted quantiles are always
            calculated.

        weighted_leaves : bool, default=False
            Weight samples inversely to the size of their leaf node.
            Only used if `weighted_quantile=True`, `max_samples_leaf!=1` and
            `aggregate_leaves_first=True`, as the samples of a leaf all have
            the same weight.

        aggregate_leaves_first : bool, default=True
            Calculate predictions using aggregated leaf values. If True, a
            single prediction is calculated over the aggregated leaf values.
            If False, a prediction is calculated for each leaf and aggregated.
            Only used if `max_samples_leaf!=1`.

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
            quantiles = "mean" if self.default_quantiles is None else self.default_quantiles
        quantiles = _check_quantiles(quantiles)

        if isinstance(interpolation, str):
            interpolation = interpolation.encode()
        elif not isinstance(interpolation, (bytes, bytearray)):
            raise ValueError(f"`interpolation` must be a string, got {interpolation!r}.")

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
            X_leaves = self._apply_validated(X)
            X_indices = None

        # Optimize for single-sample-per-leaf performance. A float of 1.0 means
        # "keep all samples" (100%), so only an integer 1 qualifies.
        if (
            isinstance(self.max_samples_leaf, (Integral, np.integer))
            and self.max_samples_leaf == 1
        ):
            y_bound_leaves = np.asarray(self.forest_.y_bound_leaves)
            y_train = np.asarray(self.forest_.y_train).T
            n_samples = X.shape[0]  # X may be sparse, so avoid len(X)
            n_trees = X_leaves.shape[1]  # the fitted trees, even if n_estimators changed
            y_pred = np.empty((n_samples, y_train.shape[1], len(quantiles)))
            for output in range(y_train.shape[1]):
                leaf_vals = np.empty((n_samples, n_trees))
                for tree in range(n_trees):
                    if X_indices is None:  # in-bag scoring
                        leaves = X_leaves[:, tree]
                        train_indices = self.forest_.first_train_indices(tree, leaves, output)
                        leaf_vals[:, tree] = np.where(
                            train_indices > 0, y_train[train_indices - 1, output], np.nan
                        )
                        if self.monotonic_cst is not None:
                            clip_min = y_bound_leaves[tree, leaves, 0]
                            clip_max = y_bound_leaves[tree, leaves, 1]
                            leaf_vals[:, tree] = np.clip(leaf_vals[:, tree], clip_min, clip_max)
                    else:  # out-of-bag scoring
                        idx_mask = X_indices[:, tree] == 1
                        leaves = X_leaves[idx_mask, tree]
                        train_indices = np.zeros(n_samples, dtype=int)
                        train_indices[idx_mask] = self.forest_.first_train_indices(
                            tree, leaves, output
                        )
                        leaf_vals[:, tree] = np.nan
                        leaf_vals[idx_mask, tree] = np.where(
                            train_indices[idx_mask] > 0,
                            y_train[train_indices[idx_mask] - 1, output],
                            np.nan,
                        )
                        if self.monotonic_cst is not None:
                            clip_min = np.full(n_samples, -np.inf)
                            clip_max = np.full(n_samples, np.inf)
                            clip_min[idx_mask] = y_bound_leaves[tree, leaves, 0]
                            clip_max[idx_mask] = y_bound_leaves[tree, leaves, 1]
                            leaf_vals[:, tree] = np.clip(leaf_vals[:, tree], clip_min, clip_max)

                # Trees where a sample is in-bag, or whose leaf is empty, have NaN values.
                has_nan = X_indices is not None or np.isnan(leaf_vals).any()
                if len(quantiles) == 1 and quantiles[0] == -1:  # calculate mean
                    func = np.nanmean if has_nan else np.mean
                    y_pred[:, output, :] = np.expand_dims(func(leaf_vals, axis=1), axis=1)
                else:  # calculate quantiles
                    method = interpolation.decode()
                    if has_nan:
                        y_q = _nanquantile_rows(leaf_vals, quantiles, method)
                    else:
                        y_q = np.quantile(leaf_vals, quantiles, method=method, axis=1)
                    y_pred[:, output, :] = y_q.T
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

        X, y = validate_data(
            self,
            X=X,
            y=y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            ensure_all_finite=False,
            # Check X against the fitted features instead of resetting them.
            reset=False,
        )

        if isinstance(kind, str):
            kind = kind.encode()
        elif not isinstance(kind, (bytes, bytearray)):
            raise ValueError(f"`kind` must be a string, got {kind!r}.")

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
            X_leaves = self._apply_validated(X)
            X_indices = None

        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        if self.monotonic_cst is not None:
            # Rank among the clipped leaf values that predict uses.
            y_ranks = self._clipped_quantile_ranks(
                y.astype(np.float64),
                X_leaves,
                X_indices,
                kind,
                aggregate_leaves_first,
            )
        else:
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

        The proximity count of an input sample and a training sample is the
        number of times that the training sample is among the samples kept in
        the leaf nodes that the input sample falls into, summed across the
        trees. With bootstrapping, each tree keeps only the samples drawn for
        it, once for each draw. Each leaf node keeps at most
        `max_samples_leaf` samples, so with `max_samples_leaf=1` (the
        default), each tree adds a count for only one training sample. To
        count all of the samples in each leaf node, fit the model with
        `max_samples_leaf=None`. See :ref:`user-guide-proximities`.

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
        proximities : list of lists of tuples, length = n_samples
            For each sample, a list of ``(index, count)`` tuples that map
            training sample indices to proximity counts.

        Notes
        -----
        For a model fit without sample weights, the proximity counts are the
        weights that `predict` gives the training samples by default. They
        differ from the proximities described by Breiman, which put all of the
        training samples down each tree, whether or not they were drawn for
        it, add one for each tree in which two samples share a leaf node, and
        divide by the number of trees:

            - https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#prox

        If `bootstrap=False` and `max_samples_leaf=None`, each proximity count
        is the number of trees in which the two samples share a leaf node.
        """
        check_is_fitted(self)
        # Check data.
        X = self._validate_X_predict(X)

        if max_proximities is None:
            max_proximities = 0
        elif not isinstance(max_proximities, Integral):
            raise ValueError(
                f"`max_proximities` must be an integer if not None, got {max_proximities}."
            )
        elif max_proximities < 1:
            raise ValueError(
                "`max_proximities` must be larger than or equal to 1 if not"
                f" None, got {max_proximities}."
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
            X_leaves = self._apply_validated(X)
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
            proximities = [list(p.items()) for p in proximities]

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

        quantiles : float or "mean", default=0.5
            The quantile that the model tries to predict. If "mean", the model
            predicts the mean.

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

    default_quantiles : float, array-like, or "mean", default=0.5
        The default quantile or list of quantiles that the model tries to
        predict. Each quantile must be in the range [0, 1]. If "mean", the
        model predicts the mean.

        .. versionadded:: 1.2

    criterion : {"squared_error", "absolute_error", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "absolute_error" for the
        mean absolute error, which minimizes the L1 loss using the median of
        each terminal node, and "poisson" which uses reduction in Poisson
        deviance to find splits.
        Training using "absolute_error" is significantly slower
        than when using "squared_error".

        .. sklearn-versionchanged:: 1.9
            Criterion `"friedman_mse"` was deprecated. It is replaced with
            `"squared_error"`, as both were always equivalent, with a warning.

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
        Leaf subsampling uses a local generator and does not change Python's
        global random state.

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

        - If None (default), then draw `X.shape[0]` samples irrespective of
          `sample_weight`.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` unweighted samples
          or `max_samples * sample_weight.sum()` weighted samples.

        .. sklearn-versionchanged:: 1.9
            Float `max_samples` is relative to `sample_weight.sum()` instead of
            `X.shape[0]` for weighted samples.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonically increasing
          - 0: no constraint
          - -1: monotonically decreasing

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multioutput regressions (i.e. when `n_outputs_ > 1`),
          - trees with multi-sample leaves (i.e. when `max_samples_leaf > 1`).

        Each tree is kept monotonic by clipping the target value that each
        leaf keeps to bounds derived from the mean targets on either side of
        each constrained split. This makes the predicted distributions much
        narrower than without constraints, so prediction intervals can be far
        too narrow. The constraints are best suited to point predictions, such
        as the median or mean. See :ref:`user-guide-fit-predict`.

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
    >>> round(qrf.score(X, y, quantiles=0.5), 2)
    0.86
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
        self.criterion = _check_criterion(criterion)
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

    default_quantiles : float, array-like, or "mean", default=0.5
        The default quantile or list of quantiles that the model tries to
        predict. Each quantile must be in the range [0, 1]. If "mean", the
        model predicts the mean.

        .. versionadded:: 1.2

    criterion : {"squared_error", "absolute_error", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "absolute_error" for the
        mean absolute error, which minimizes the L1 loss using the median of
        each terminal node, and "poisson" which uses reduction in Poisson
        deviance to find splits.
        Training using "absolute_error" is significantly slower
        than when using "squared_error".

        .. sklearn-versionchanged:: 1.9
            Criterion `"friedman_mse"` was deprecated. It is replaced with
            `"squared_error"`, as both were always equivalent, with a warning.

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

        Leaf subsampling uses a local generator and does not change Python's
        global random state.

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

        - If None (default), then draw `X.shape[0]` samples irrespective of
          `sample_weight`.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` unweighted samples
          or `max_samples * sample_weight.sum()` weighted samples.

        .. sklearn-versionchanged:: 1.9
            Float `max_samples` is relative to `sample_weight.sum()` instead of
            `X.shape[0]` for weighted samples.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonically increasing
          - 0: no constraint
          - -1: monotonically decreasing

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multioutput regressions (i.e. when `n_outputs_ > 1`),
          - trees with multi-sample leaves (i.e. when `max_samples_leaf > 1`).

        Each tree is kept monotonic by clipping the target value that each
        leaf keeps to bounds derived from the mean targets on either side of
        each constrained split. This makes the predicted distributions much
        narrower than without constraints, so prediction intervals can be far
        too narrow. The constraints are best suited to point predictions, such
        as the median or mean. See :ref:`user-guide-fit-predict`.

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
    >>> round(qrf.score(X, y, quantiles=0.5), 2)
    0.89
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
        self.criterion = _check_criterion(criterion)
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
