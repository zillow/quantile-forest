# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

from libc.math cimport ceil, fabs, floor, round
from libc.string cimport memset
from libcpp.algorithm cimport sort as sort_cpp
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue
from libcpp.set cimport set
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np
np.import_array()

from scipy import sparse


cdef inline void parallel_qsort_asc(
    vector[double]& a,
    vector[double]& b,
    int left,
    int right,
) noexcept nogil:
    """Sort lists `a` and `b` in ascending order by `a`."""
    cdef int i, j
    cdef double pivot

    i = left
    j = right
    pivot = a[(left + right) / 2]

    while True:
        while a[i] < pivot:
            i += 1
        while pivot < a[j]:
            j -= 1
        if i <= j:
            a[i], a[j] = a[j], a[i]
            b[i], b[j] = b[j], b[i]
            i += 1
            j -= 1
        if i > j:
            break

    if left < j:
        parallel_qsort_asc(a, b, left, j)
    if i < right:
        parallel_qsort_asc(a, b, i, right)


cpdef double calc_mean(vector[double]& inputs) noexcept nogil:
    """Return the mean for list of inputs.

    Parameters
    ----------
    inputs : list of floats
        List of inputs from which to compute mean.

    Returns
    -------
    out : float
        Mean for `inputs` as float.
    """
    cdef SIZE_t n_inputs
    cdef SIZE_t i
    cdef double out = 0

    n_inputs = inputs.size()

    if n_inputs < 1:
        return out

    for i in range(n_inputs):
        out += inputs[i]
    out /= <double>n_inputs

    return out


cpdef double calc_weighted_mean(
    vector[double]& inputs,
    vector[double]& weights,
) noexcept nogil:
    """Return the mean for list of weighted inputs.

    Parameters
    ----------
    inputs : list of floats
        List of inputs from which to compute mean.

    weights : list of ints
        List of weights for `inputs`.

    Returns
    -------
    out : float
        Mean for `inputs` as float.
    """
    cdef SIZE_t n_inputs
    cdef SIZE_t i
    cdef double weight, cum_weights
    cdef double out = 0

    n_inputs = inputs.size()

    if n_inputs < 1:
        return out

    cum_weights = 0
    for i in range(n_inputs):
        weight = weights[i]
        if weight > 0:
            out += inputs[i] * <double>weight
            cum_weights += weight
    out /= <double>cum_weights

    return out


cpdef vector[double] calc_quantile(
    vector[double]& inputs,
    vector[double] quantiles,
    char* interpolation=b"linear",
    bint issorted=<bint>False,
) noexcept nogil:
    """Return quantiles for list of inputs.

    A desired quantile is calculated from the input rank ``x`` such that
    ``x = (N + 1 - 2C)q + C``, where ``q`` is the quantile, ``N`` is the
    number of samples, and ``C`` is a constant (degree of freedom).

    If the desired quantile lies between two input ranks ``i < j``, then the
    quantile is interpolated based on the specified `interpolation` method.

    Parameters
    ----------
    inputs : list of floats
        List of inputs from which to compute quantiles.

    quantiles : list of floats
        Quantiles in the range [0, 1] to compute.

    interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}, \
            default="linear"
        Specifies the interpolation method to use when the desired quantile
        lies between two data points ``i < j``:

        - If "'linear", then ``i + (j - i) * fraction``, where ``fraction`` is
          the fractional part of the index surrounded by ``i`` and ``j``.
        - If "lower", then ``i``.
        - If "higher", then ``j``.
        - If "nearest", then ``i`` or ``j``, whichever is nearest.
        - If "midpoint", then ``(i + j) / 2``.

    issorted : bool, default=False
        Inputs are sorted in ascending order.

    Returns
    -------
    out : array-like of shape (n_quantiles)
        Quantiles for `inputs` as floats. Empty list if empty inputs.
    """
    cdef double C = 1

    cdef SIZE_t n_inputs, n_quantiles
    cdef string s_interpolation
    cdef SIZE_t i
    cdef double f
    cdef double quantile
    cdef double idx, v_floor, v_ceil, frac
    cdef vector[double] out

    n_inputs = inputs.size()
    n_quantiles = quantiles.size()

    s_interpolation = <string>interpolation

    if n_inputs < 1:
        return out
    elif n_inputs == 1:
        out = vector[double](n_quantiles)
        for i in range(n_quantiles):
            out[i] = inputs[0]
        return out

    if not issorted:
        sort_cpp(inputs.begin(), inputs.end())

    f = n_inputs + 1 - (2 * C)

    out = vector[double](n_quantiles)

    for i in range(n_quantiles):
        quantile = quantiles[i]

        # Calculate the quantile's (potentially fractional) index.
        idx = quantile * f + C - 1

        # Check if the quantile is the first or last value.
        if idx >= n_inputs - 1:
            out[i] = inputs[n_inputs - 1]
            continue
        if idx <= 0:
            out[i] = inputs[0]
            continue

        v_floor = inputs[<int>floor(idx)]
        v_ceil = inputs[<int>ceil(idx)]

        # Check if the quantile does not lie between two values.
        if v_floor == v_ceil:
            out[i] = v_floor
            continue

        # Calculate the fractional remainder.
        frac = idx % (<int>floor(idx)) if idx >= 1 else idx

        # Interpolate the quantile, as it lies between two values.
        if s_interpolation == <char*>b"lower":
            out[i] = v_floor
        elif s_interpolation == <char*>b"higher":
            out[i] = v_ceil
        elif s_interpolation == <char*>b"midpoint":
            out[i] = 0.5 * (v_floor + v_ceil)
        elif s_interpolation == <char*>b"nearest":
            if fabs(frac - 0.5) < 1e-16:
                out[i] = inputs[<int>(round(idx / 2) * 2)]
            else:
                out[i] = v_floor if frac < 0.5 else v_ceil
        elif s_interpolation == <char*>b"linear":
            out[i] = v_floor + frac * (v_ceil - v_floor)

    return out


cpdef vector[double] calc_weighted_quantile(
    vector[double]& inputs,
    vector[double]& weights,
    vector[double] quantiles,
    char* interpolation=b"linear",
    bint issorted=<bint>False,
) noexcept nogil:
    """Return quantiles for list of weighted inputs.

    The weighted quantile is calculated by first calculating the empirical
    cumulative distribution function (ECDF) from the weights and then finding
    the first index where the proportion of the ECDF weight is greater than
    the quantile's proportion of total weight.

    If the weights represent frequencies of appearance ``f_v`` for each value
    ``v``, the output is equivalent to calculating the unweighted quantile
    with each value ``v`` appearing ``f_v`` times.

    A desired quantile is calculated from the input rank ``x`` such that
    ``x = (N + 1 - 2C)q + C``, where ``q`` is the quantile, ``N`` is the
    number of samples, and ``C`` is a constant (degree of freedom).

    If the desired quantile lies between two input ranks ``i < j``, then the
    quantile is interpolated based on the specified `interpolation` method.

    Parameters
    ----------
    inputs : list of floats
        List of inputs from which to compute quantiles.

    weights : list of ints
        List of weights for `inputs`.

    quantiles : list of floats
        Quantiles in the range [0, 1] to compute.

    interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}, \
            default="linear"
        Specifies the interpolation method to use when the desired quantile
        lies between two data points ``i < j``:

        - If "linear", then ``i + (j - i) * fraction``, where ``fraction`` is
          the fractional part of the index surrounded by ``i`` and ``j``.
        - If "lower", then ``i``.
        - If "higher", then ``j``.
        - If "nearest", then ``i`` or ``j``, whichever is nearest.
        - If "midpoint", then ``(i + j) / 2``.

    issorted : bool, default=False
        Inputs and weights are sorted by inputs in ascending order.

    Returns
    -------
    out : array-like of shape (n_quantiles)
        Quantiles for `inputs` as floats. Empty list if empty inputs.
    """
    cdef double C = 1

    cdef SIZE_t n_inputs, n_quantiles
    cdef string s_interpolation
    cdef SIZE_t i, j
    cdef double f
    cdef double quantile
    cdef vector[double] cum_weights, sorted_quantile_indices
    cdef int idx_floor, idx_ceil
    cdef double p, p_floor, p_ceil
    cdef double v_floor, v_ceil, frac
    cdef vector[double] out

    n_inputs = inputs.size()
    n_quantiles = quantiles.size()

    s_interpolation = <string>interpolation

    if n_inputs < 1:
        return out

    if not issorted:
        parallel_qsort_asc(inputs, weights, 0, n_inputs - 1)

    cum_weights = vector[double](n_inputs)

    # Calculate the empirical cumulative distribution function (ECDF).
    cum_weights[0] = weights[0]
    for i in range(1, n_inputs):
        cum_weights[i] = cum_weights[i - 1] + weights[i]

    if cum_weights[n_inputs - 1] <= 0:
        return out

    f = cum_weights[n_inputs - 1] + 1 - (2 * C)

    # Get the indices that would sort the quantiles in ascending order.
    sorted_quantile_indices = vector[double](n_quantiles)
    for i in range(<SIZE_t>(sorted_quantile_indices.size())):
        sorted_quantile_indices[i] = <double>i
    parallel_qsort_asc(quantiles, sorted_quantile_indices, 0, n_quantiles - 1)

    out = vector[double](n_quantiles)

    idx_floor = 0
    idx_ceil = 1

    for i in range(n_quantiles):
        quantile = quantiles[i]

        # Assign the output based on the input quantile ordering.
        i = <SIZE_t>sorted_quantile_indices[i]

        # Calculate the quantile's proportion of total weight.
        p = quantile * f + C

        # Find the first index where the proportion of weight exceeds p.
        for j in range(idx_floor, n_inputs):
            if p >= cum_weights[j]:
                if weights[j] > 0:
                    idx_floor = j
                    idx_ceil = j + 1
            else:
                break

        # Ensure `idx_ceil` is not out of bounds.
        if idx_ceil > n_inputs - 1:
            idx_ceil = n_inputs - 1

        # Ensure the input indexed by `idx_ceil` does not have 0 weight.
        while weights[idx_ceil] == 0:
            idx_ceil = idx_ceil + 1
            if idx_ceil > n_inputs - 1:
                idx_ceil = idx_floor
                break

        # Check if the quantile is the first, last, or known index value.
        if idx_floor >= n_inputs - 1:
            out[i] = inputs[n_inputs - 1]
            continue
        if idx_ceil <= 0:
            out[i] = inputs[0]
            continue
        if idx_floor == idx_ceil:
            out[i] = inputs[idx_floor]
            continue

        v_floor = inputs[idx_floor]
        v_ceil = inputs[idx_ceil]

        # Check if the quantile does not lie between two values.
        if v_floor == v_ceil:
            out[i] = v_floor
            continue

        # Calculate the proportion of weights at indices from the ECDF.
        p_floor = (cum_weights[idx_floor] - C) / f
        p_ceil = (cum_weights[idx_ceil] - C) / f

        # Ensure `p_floor` and `p_ceil` are not out of bounds.
        p_floor = min(quantile, p_floor)
        p_ceil = max(quantile, p_ceil)

        # Interpolate the minimum proportion for `p_ceil` (for weights > 1).
        # Equivalent to taking the index of the first occurrence of `v_ceil`.
        p_ceil = p_floor + ((p_ceil - p_floor) / weights[idx_ceil])

        # Check if the quantile lies at the interpolated ceiling value.
        if quantile >= p_ceil:
            out[i] = v_ceil
            continue

        # Calculate the fractional remainder.
        frac = (quantile - p_floor) / (p_ceil - p_floor)

        # Check if the quantile lies at the ceiling or floor values.
        if frac <= 0:
            out[i] = v_floor
            continue
        elif frac >= 1:
            out[i] = v_ceil
            continue

        # Interpolate the quantile, as it lies between two values.
        if s_interpolation == <char*>b"lower":
            out[i] = v_floor
        elif s_interpolation == <char*>b"higher":
            out[i] = v_ceil
        elif s_interpolation == <char*>b"midpoint":
            out[i] = 0.5 * (v_floor + v_ceil)
        elif s_interpolation == <char*>b"nearest":
            if fabs(frac - 0.5) < 1e-16:
                out[i] = v_floor if cum_weights[idx_floor] % 2 else v_ceil
            else:
                out[i] = v_floor if frac < 0.5 else v_ceil
        elif s_interpolation == <char*>b"linear":
            out[i] = v_floor + frac * (v_ceil - v_floor)

    return out


cpdef double calc_quantile_rank(
    vector[double]& inputs,
    double score,
    char* kind=b"rank",
    bint issorted=<bint>False,
) noexcept nogil:
    """Return quantile rank of score relative to inputs.

    Parameters
    ----------
    inputs : list of floats
        List of inputs with which `score` is compared.

    score : float
        Value for which to get quantile rank.

    kind : {"rank", "weak", "strict", "mean"}, default="rank"
        Specifies the interpretation of the resulting score:

        - If "rank", then average percentage ranking of score. If multiple
          matches, average the percentage rankings of all matching scores.
        - If "weak", then only values that are less than or equal to the
          provided score are counted. Corresponds to the definition of a
          cumulative distribution function.
        - If "strict", then similar to "weak", except that only values that
          are strictly less than the provided score are counted.
        - If "mean", then the average of the "weak" and "strict" scores.

    issorted : bool, default=False
        Inputs are sorted in ascending order.

    Returns
    -------
    out : float
        Quantile rank for `score` as float. -1 if empty inputs.
    """
    cdef SIZE_t n_inputs
    cdef string s_kind
    cdef SIZE_t i
    cdef int left, right
    cdef double out = -1

    n_inputs = inputs.size()

    s_kind = <string>kind

    if n_inputs < 1:
        return out

    if not issorted:
        sort_cpp(inputs.begin(), inputs.end())

    left = 0
    right = 0
    for i in range(n_inputs):
        if inputs[i] < score:
            left = right = i + 1
        elif inputs[i] == score:
            right = i + 1
        else:
            break

    if s_kind == <char*>b"rank":
        out = (right + left + (1 if right > left else 0)) * 0.5 / n_inputs
    elif s_kind == <char*>b"weak":
        out = right / (<double>n_inputs)
    elif s_kind == <char*>b"strict":
        out = left / (<double>n_inputs)
    elif s_kind == <char*>b"mean":
        out = (left + right) * 0.5 / n_inputs

    return out


cpdef vector[SIZE_t] generate_unsampled_indices(
    vector[SIZE_t] sample_indices,
    vector[set[SIZE_t]] duplicates,
) noexcept nogil:
    """Return a list of every unsampled index, accounting for duplicates.

    Parameters
    ----------
    sample_indices : array-like of shape (n_samples)
        Sample indices for which to get duplicates.

    duplicates : list of sets
        List of sets of functionally identical indices.

    Returns
    -------
    unsampled_indices : array-like
        List of unsampled indices.
    """
    cdef SIZE_t i
    cdef SIZE_t sampled_idx
    cdef set[SIZE_t] sampled_set
    cdef SIZE_t n_samples, n_duplicates
    cdef vector[SIZE_t] unsampled_indices

    n_samples = sample_indices.size()
    n_duplicates = duplicates.size()

    for i in range(n_samples):
        sampled_set.insert(sample_indices[i])

    # Account for duplicates of sampled indices.
    for i in range(n_duplicates):
        for sampled_idx in duplicates[i]:
            if sampled_set.count(sampled_idx):
                sampled_set.insert(duplicates[i].begin(), duplicates[i].end())

    # If the index is not in `sampled_set`, it is unsampled.
    for i in range(n_samples):
        if not sampled_set.count(i):
            unsampled_indices.push_back(i)

    return unsampled_indices


cdef class QuantileForest:
    """Representation of a quantile forest.

    Attributes
    ----------
    y_train : array-like of shape (n_samples, n_outputs)
        Training target values. Assumes values are sorted in ascending order.

    y_train_leaves : array-like of shape \
            (n_estimators, n_leaves, n_outputs, n_indices)
        List of trees, each with a list of nodes, each with a list of indices
        of the training samples residing at that node. Nodes with no samples
        (e.g., internal nodes) are empty. Internal nodes are included so that
        leaf node indices match their ``est.apply`` outputs. Each node list is
        padded to equal length with 0s.

    sparse_pickle : bool, default=False
        Pickle using a SciPy sparse matrix.
    """

    def __cinit__(
        self,
        np.ndarray[DOUBLE_t, ndim=2] y_train,
        np.ndarray[SIZE_t, ndim=4] y_train_leaves,
        bint sparse_pickle=<bint>False,
    ):
        """Constructor."""
        self.y_train = y_train
        self.y_train_leaves = y_train_leaves
        self.sparse_pickle = sparse_pickle

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        if self.sparse_pickle:
            y_train_leaves = np.empty(shape=(0, 0, 0, 0), dtype=np.int64)
            kwargs = {"y_train_leaves": np.asarray(self.y_train_leaves)}
        else:
            y_train_leaves = np.asarray(self.y_train_leaves)
            kwargs = {}
        args = (np.asarray(self.y_train), y_train_leaves, self.sparse_pickle)
        return (QuantileForest, args, self.__getstate__(**kwargs))

    def __getstate__(self, **kwargs):
        """Getstate re-implementation, for pickling."""
        d = {}
        if self.sparse_pickle:
            matrix = kwargs["y_train_leaves"]
            reshape = (matrix.shape[2], matrix.shape[0] * matrix.shape[1] * matrix.shape[2])
            d["shape"] = matrix.shape
            d["matrix"] = sparse.csc_matrix(matrix.reshape(reshape))
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        if self.sparse_pickle:
            self.y_train_leaves = d["matrix"].toarray().reshape(d["shape"])

    cpdef np.ndarray predict(
        self,
        vector[double] quantiles,
        SIZE_t[:, :] X_leaves,
        UINT8_t[:, :] X_indices=None,
        char* interpolation=b"linear",
        bint weighted_quantile=<bint>True,
        bint weighted_leaves=<bint>False,
        bint aggregate_leaves_first=<bint>True,
    ):
        """Return predictions for ``est.apply`` outputs.

        Parameters
        ----------
        quantiles : list of floats
            Quantiles the in range [0, 1] to compute. -1 to compute mean.

        X_leaves : array-like of shape (n_samples, n_estimators)
            Target leaf node indices along samples and trees.

        X_indices : array-like of shape (n_samples, n_estimators), \
                default=None
            For each sample, the list of base estimators used. For generating
            OOB leaf members, it is the list of training samples excluded from
            the bootstrapping process. By default, selects all samples across
            all base estimators. 1 if sample should be selected, 0 otherwise.

        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}, \
                default="linear"
            The interpolation method to use when calculating quantiles.

        weighted_quantile : bool, default=True
            Calculate weighted quantiles.

        weighted_leaves : bool, default=False
            Weight samples inversely to the size of their leaf node.

        aggregate_leaves_first : bool, default=True
            Calculate predictions using leaf values aggregated across trees.

        Returns
        -------
        preds : array-like of shape (n_samples, n_quantiles, n_outputs)
            Quantiles or means for samples as floats.
        """
        cdef vector[double] median = [0.5]

        cdef SIZE_t n_quantiles, n_samples, n_trees, n_outputs, n_train
        cdef SIZE_t i, j, k, l
        cdef bint use_mean
        cdef vector[double] leaf_samples
        cdef vector[double] leaf_weights
        cdef vector[vector[SIZE_t]] train_indices
        cdef vector[vector[double]] train_weights
        cdef SIZE_t idx, train_idx
        cdef double train_wgt
        cdef vector[int] n_leaf_samples
        cdef int n_total_samples, n_total_trees
        cdef double train_weight
        cdef vector[vector[double]] leaf_preds
        cdef vector[double] pred
        cdef np.ndarray[DOUBLE_t, ndim=3] preds
        cdef double[:, :, :] preds_view

        n_quantiles = len(quantiles)
        n_samples = X_leaves.shape[0]
        n_trees = X_leaves.shape[1]

        n_outputs = self.y_train.size()
        n_train = self.y_train[0].size()
        max_idx = self.y_train_leaves.shape[3]

        use_mean = False
        if len(quantiles) == 1 and quantiles[0] == -1:
            use_mean = True
        else:
            for q in quantiles:
                if q < 0 or q > 1:
                    raise ValueError(f"Quantiles must be in the range [0, 1], got {q}.")

        if X_indices is not None:
            if X_indices.shape[1] != X_leaves.shape[1]:
                raise ValueError(
                    f"X_indices.shape[1]={X_indices.shape[1]} must equal "
                    f"X_leaves.shape[1]={X_leaves.shape[1]}, or X_indices "
                    "must be None."
                )

        interps = [b"linear", b"lower", b"higher", b"midpoint", b"nearest"]
        if interpolation not in interps:
            raise ValueError(f"Invalid interpolation method {interpolation}.")

        # Initialize NumPy array with NaN values and get view for nogil.
        preds = np.full((n_samples, n_quantiles, n_outputs), np.nan, dtype=np.float64)
        preds_view = preds  # memoryview

        with nogil:
            idx = 1 if aggregate_leaves_first else n_trees
            train_indices = vector[vector[SIZE_t]](idx)

            n_leaf_samples = vector[int](n_trees)
            leaf_preds = vector[vector[double]](n_quantiles)
            for i in range(<SIZE_t>(leaf_preds.size())):
                leaf_preds[i].reserve(idx)

            if weighted_quantile:
                train_weights = vector[vector[double]](idx)
                leaf_weights = vector[double](n_train)

            for i in range(n_samples):
                n_total_samples = 0
                n_total_trees = 0
                for j in range(n_trees):
                    if X_indices is None or X_indices[i, j] is True:
                        n_leaf_samples[j] = 0
                        for k in range(max_idx):
                            if self.y_train_leaves[j, X_leaves[i, j], 0, k] != 0:
                                n_leaf_samples[j] += 1
                        n_total_samples += n_leaf_samples[j]
                        n_total_trees += 1

                for j in range(n_outputs):
                    for k in range(<SIZE_t>(train_indices.size())):
                        train_indices[k].clear()
                    for k in range(<SIZE_t>(leaf_preds.size())):
                        leaf_preds[k].clear()

                    # Accumulate training indices across leaves for each tree.
                    # If `aggregate_leaves_first`, also accumulate across trees.
                    for k in range(n_trees):
                        if X_indices is None or X_indices[i, k] is True:
                            idx = 0 if aggregate_leaves_first else k
                            train_indices[idx].insert(
                                train_indices[idx].end(),
                                &self.y_train_leaves[k, X_leaves[i, k], j, 0],
                                &self.y_train_leaves[k, X_leaves[i, k], j, max_idx],
                            )

                    if weighted_quantile:
                        for k in range(<SIZE_t>(train_weights.size())):
                            train_weights[k].clear()
                        for k in range(n_trees):
                            if X_indices is None or X_indices[i, k] is True:
                                idx = 0 if aggregate_leaves_first else k
                                train_weight = 1
                                if weighted_leaves:
                                    train_weight = 0
                                    if n_leaf_samples[k] > 0:
                                        train_weight = 1 / <double>n_leaf_samples[k]
                                        train_weight *= <double>n_total_samples
                                        train_weight /= <double>n_total_trees
                                train_weights[idx].insert(
                                    train_weights[idx].end(), max_idx, train_weight
                                )

                        # For each list of training indices, calculate output.
                        for k in range(<SIZE_t>(train_indices.size())):
                            if train_indices[k].size() == 0:
                                continue

                            # Reset leaf weights for all training indices to 0.
                            memset(&leaf_weights[0], 0, n_train * sizeof(double))

                            # Sum the weights/counts for each training index.
                            for l in range(<SIZE_t>(train_indices[k].size())):
                                train_idx = train_indices[k][l]
                                train_wgt = train_weights[k][l]
                                if train_idx != 0:
                                    leaf_weights[train_idx - 1] += train_wgt

                            # Calculate quantiles (or mean).
                            if not use_mean:
                                pred = calc_weighted_quantile(
                                    self.y_train[j],
                                    leaf_weights,
                                    quantiles,
                                    interpolation,
                                    issorted=True,
                                )
                                for l in range(<SIZE_t>(pred.size())):
                                    leaf_preds[l].push_back(pred[l])
                            else:
                                if self.y_train[j].size() > 0:
                                    pred = vector[double](1)
                                    pred[0] = calc_weighted_mean(self.y_train[j], leaf_weights)
                                    leaf_preds[0].push_back(pred[0])
                    else:
                        # For each list of training indices, calculate output.
                        for k in range(<SIZE_t>(train_indices.size())):
                            if train_indices[k].size() == 0:
                                continue

                            # Clear list of training target values.
                            leaf_samples.clear()

                            # Get training target values associated with indices.
                            for train_idx in train_indices[k]:
                                if train_idx != 0:
                                    leaf_samples.push_back(self.y_train[j][train_idx - 1])

                            # Calculate quantiles (or mean).
                            if not use_mean:
                                pred = calc_quantile(
                                    leaf_samples,
                                    quantiles,
                                    interpolation,
                                    issorted=False,
                                )
                                for l in range(<SIZE_t>(pred.size())):
                                    leaf_preds[l].push_back(pred[l])
                            else:
                                if leaf_samples.size() > 0:
                                    pred = vector[double](1)
                                    pred[0] = calc_mean(leaf_samples)
                                    leaf_preds[0].push_back(pred[0])

                    # Average the quantile predictions across accumulations.
                    if not use_mean:
                        for k in range(<SIZE_t>(leaf_preds.size())):
                            if leaf_preds[k].size() == 1:
                                preds_view[i, k, j] = leaf_preds[k][0]
                            elif leaf_preds[k].size() > 1:
                                pred = calc_quantile(
                                    leaf_preds[k],
                                    median,
                                    interpolation,
                                    issorted=False,
                                )
                                preds_view[i, k, j] = pred[0]
                    else:
                        if leaf_preds[0].size() == 1:
                            preds_view[i, 0, j] = leaf_preds[0][0]
                        elif leaf_preds[0].size() > 1:
                            preds_view[i, 0, j] = calc_mean(leaf_preds[0])

        return np.asarray(preds_view)

    cpdef np.ndarray quantile_ranks(
        self,
        double[:, :] y_scores,
        SIZE_t[:, :] X_leaves,
        UINT8_t[:, :] X_indices=None,
        char* kind=b"rank",
        bint aggregate_leaves_first=<bint>True,
    ):
        """Return quantile ranks for ``est.apply`` outputs with scores.

        Parameters
        ----------
        y_scores : array-like of shape (n_samples, n_outputs)
            Target values for which to calculate quantile ranks.

        X_leaves : array-like of shape (n_samples, n_estimators)
            Target leaf node indices along samples and trees.

        X_indices : array-like of shape (n_samples, n_estimators), \
                default=None
            For each sample, the list of base estimators used. For generating
            OOB leaf members, this is the list of training samples excluded
            from the bootstrap process. By default, selects all samples across
            all base estimators. 1 if sample should be selected, 0 otherwise.

        kind : {"rank", "weak", "strict", "mean"}, default="rank"
            Specifies the interpretation of the resulting score.

        aggregate_leaves_first : bool, default=True
            Calculate ranks using using leaf values aggregated across trees.

        Returns
        -------
        ranks : array-like of shape (n_samples, n_outputs)
            Quantiles ranks in range [0, 1] for samples as floats.
        """
        cdef SIZE_t n_samples, n_trees, n_outputs
        cdef SIZE_t i, j
        cdef vector[double] leaf_samples
        cdef vector[vector[SIZE_t]] train_indices
        cdef SIZE_t idx, train_idx
        cdef vector[double] leaf_preds
        cdef double pred
        cdef np.ndarray[DOUBLE_t, ndim=2] ranks
        cdef double[:, :] ranks_view

        n_outputs = y_scores.shape[0]

        n_samples = X_leaves.shape[0]
        n_trees = X_leaves.shape[1]

        max_idx = self.y_train_leaves.shape[3]

        if X_indices is not None:
            if X_indices.shape[1] != X_leaves.shape[1]:
                raise ValueError(
                    f"X_indices.shape[1]={X_indices.shape[1]} must equal "
                    f"X_leaves.shape[1]={X_leaves.shape[1]}, or X_indices "
                    "must be None."
                )

        kinds = [b"rank", b"weak", b"strict", b"mean"]
        if kind not in kinds:
            raise ValueError(f"Invalid kind {kind}.")

        # Initialize NumPy array with NaN values and get view for nogil.
        ranks = np.full((n_samples, n_outputs), np.nan, dtype=np.float64)
        ranks_view = ranks  # memoryview

        with nogil:
            idx = 1 if aggregate_leaves_first else n_trees
            train_indices = vector[vector[SIZE_t]](idx)

            for i in range(n_samples):
                for j in range(n_outputs):
                    for k in range(<SIZE_t>(train_indices.size())):
                        train_indices[k].clear()
                    leaf_samples.clear()
                    leaf_preds.clear()

                    # Accumulate training indices across leaves for each tree.
                    # If `aggregate_leaves_first`, also accumulate across trees.
                    for k in range(n_trees):
                        if X_indices is None or X_indices[i, k] is True:
                            idx = 0 if aggregate_leaves_first else k
                            train_indices[idx].insert(
                                train_indices[idx].end(),
                                &self.y_train_leaves[k, X_leaves[i, k], j, 0],
                                &self.y_train_leaves[k, X_leaves[i, k], j, max_idx],
                            )

                    # For each list of training indices, calculate rank.
                    for k in range(<SIZE_t>(train_indices.size())):
                        if train_indices[k].size() == 0:
                            continue

                        # Get training target values associated with indices.
                        for train_idx in train_indices[k]:
                            if train_idx != 0:
                                leaf_samples.push_back(self.y_train[j][train_idx - 1])

                        # Calculate rank.
                        pred = calc_quantile_rank(leaf_samples, y_scores[j, i], kind=kind)
                        if pred != -1:
                            leaf_preds.push_back(pred)

                    # Average the rank predictions across accumulations.
                    if leaf_preds.size() == 1:
                        ranks_view[i, j] = leaf_preds[0]
                    elif leaf_preds.size() > 1:
                        ranks_view[i, j] = calc_mean(leaf_preds)

        return np.asarray(ranks_view)

    cpdef vector[map[SIZE_t, SIZE_t]] proximity_counts(
        self,
        SIZE_t[:, :] X_leaves,
        UINT8_t[:, :] X_indices=None,
        UINT32_t max_proximities=0,
        SIZE_t[:, :] sorter=None,
    ):
        """Return proximity counts of the training samples for target leaves.

        For each tree, iterates through each scoring sample's leaf and counts
        occurrence (proximity) of training samples residing in the same node.

        Parameters
        ----------
        X_leaves : array-like of shape (n_samples, n_estimators)
            Target leaf node indices along samples and trees.

        X_indices : array-like of shape (n_samples, n_estimators), \
                default=None
            For each sample, the list of base estimators used. For generating
            OOB leaf members, this is the list of training samples excluded
            from the bootstrap process. By default, selects all samples across
            all base estimators. 1 if sample should be selected, 0 otherwise.

        max_proximities : int, default=0
            Maximum number of proximities to return for each scoring sample,
            prioritized by proximity count. By default, return all proximity
            counts for each sample.

        sorter : array-like of shape (n_train, n_outputs), default=None
            The indices that would sort the target values in ascending order.
            Used to associate ``est.apply`` outputs with sorted target values.

        Returns
        -------
        proximities : list of dicts
            Dicts mapping sample indices to proximity counts.
        """
        cdef vector[map[SIZE_t, SIZE_t]] proximities
        cdef SIZE_t n_samples, n_trees, n_train
        cdef SIZE_t i, j
        cdef vector[SIZE_t] train_indices
        cdef vector[int] leaf_weights
        cdef SIZE_t train_idx
        cdef int cutoff, train_wgt
        cdef priority_queue[pair[int, SIZE_t]] queue
        cdef pair[int, SIZE_t] entry

        n_samples = X_leaves.shape[0]
        n_trees = X_leaves.shape[1]

        n_train = self.y_train[0].size()
        max_idx = self.y_train_leaves.shape[3]

        if X_indices is not None:
            if X_indices.shape[1] != X_leaves.shape[1]:
                raise ValueError(
                    f"X_indices.shape[1]={X_indices.shape[1]} must equal "
                    f"X_leaves.shape[1]={X_leaves.shape[1]}, or X_indices "
                    "must be None."
                )

        if max_proximities < 1 or n_train < max_proximities:
            max_proximities = n_train

        # Initialize map of proximity counts for every target leaf node.
        proximities = vector[map[SIZE_t, SIZE_t]](n_samples)

        with nogil:
            leaf_weights = vector[int](n_train)
            for i in range(n_samples):
                train_indices.clear()
                memset(&leaf_weights[0], 0, n_train * sizeof(int))

                # Accumulate training indices across leaves for each tree.
                for j in range(n_trees):
                    if X_indices is None or X_indices[i, j] is True:
                        train_indices.insert(
                            train_indices.end(),
                            &self.y_train_leaves[j, X_leaves[i, j], 0, 0],
                            &self.y_train_leaves[j, X_leaves[i, j], 0, max_idx],
                        )

                # Sum the weights/counts for each training index.
                for train_idx in train_indices:
                    if train_idx != 0:
                        train_idx -= 1
                        if sorter is not None:
                            # Align to the input (unsorted) training index.
                            train_idx = sorter[:, 0][train_idx]
                        leaf_weights[train_idx] += 1

                # Build priority queue by -weight (so smallest weight is top).
                # Using a priority queue can avoid having to sort all samples.
                cutoff = 1
                for j in range(n_train):
                    train_idx = j
                    train_wgt = leaf_weights[j]
                    if train_wgt >= cutoff:
                        queue.push(pair[int, SIZE_t](-train_wgt, train_idx))
                        if queue.size() > max_proximities:
                            queue.pop()  # remove the top (smallest) weight
                            cutoff = -queue.top().first  # set the new cutoff

                # Build map of proximity indices and counts.
                while not queue.empty():
                    entry = queue.top()
                    proximities[i][entry.second] = -entry.first
                    queue.pop()

        return proximities
