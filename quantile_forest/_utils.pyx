from libc.stdlib cimport free, malloc
from libcpp.set cimport set
from libcpp.vector cimport vector

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef inline void parallel_qsort_asc(
    vector[double]& a,
    vector[double]& b,
    intp_t left,
    intp_t right,
) noexcept nogil:
    """Sort two lists `a` and `b` in parallel in ascending order by `a`.

    The sorting is performed in-place on `a`, and the elements of `b` are
    rearranged to maintain the same index correspondence with `a`.

    The sorting is performed using quicksort.

    Parameters
    ----------
    a : list of floats
        Primary list to be sorted in ascending order.
    b : list of floats
        Secondary list, whose elements are rearranged in parallel with `a`.
    left : int
        Starting index of the current subarray to sort.
    right : int
        Ending index of the current subarray to sort.
    """
    cdef intp_t i, j
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


cpdef vector[intp_t] generate_unsampled_indices(
    vector[intp_t] sample_indices,
    intp_t n_total_samples,
    vector[set[intp_t]] duplicates,
) noexcept nogil:
    """Return a list of every unsampled index, accounting for duplicates.

    Parameters
    ----------
    sample_indices : array-like of shape (n_samples,)
        Sampled indices.

    n_total_samples : int
        Number of total samples, sampled and unsampled.

    duplicates : list of sets of ints
        List of sets of functionally identical indices.

    Returns
    -------
    unsampled_indices : array-like
        List of unsampled indices.
    """
    cdef intp_t n_samples, n_duplicates
    cdef intp_t i
    cdef intp_t sampled_idx
    cdef set[intp_t] sampled_set
    cdef vector[intp_t] unsampled_indices

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
    for i in range(n_total_samples):
        if not sampled_set.count(i):
            unsampled_indices.push_back(i)

    return unsampled_indices


cpdef group_indices_by_value(cnp.ndarray[intp_t, ndim=1] a):
    """Group indices of a sorted array based on unique values.

    Parameters
    ----------
    a : array-like of shape (n_samples,)
        Input array. The array is expected to contain integers, and the
        function will group the indices of elements with the same value.

    Returns
    -------
    np_unq_items : array-like
        A NumPy array containing the unique values from the input array `a`,
        sorted in ascending order.

    unq_idx : list of array-like
        A list of NumPy arrays, where each array contains the indices of the
        input array `a` corresponding to each unique value in `np_unq_items`.
        The indices are sorted based on the original order in `a`.
    """
    cdef intp_t num_samples
    cdef intp_t i
    cdef cnp.ndarray[intp_t, ndim=1] sort_idx
    cdef cnp.ndarray[intp_t, ndim=1] a_sorted
    cdef intp_t prev_value
    cdef intp_t count, unq_count_idx
    cdef intp_t* unq_count
    cdef bint* unq_first
    cdef intp_t* unq_first_indices

    num_samples = a.shape[0]
    sort_idx = np.argsort(a)
    a_sorted = a[sort_idx]
    unq_count_idx = 0
    unq_count = <intp_t*>malloc(num_samples * sizeof(intp_t))
    unq_first = <bint*>malloc(num_samples * sizeof(bint))
    unq_first_indices = <intp_t*>malloc(num_samples * sizeof(intp_t))

    if unq_count == NULL or unq_first == NULL or unq_first_indices == NULL:
        raise MemoryError("Memory allocation failed.")

    with nogil:
        # Initialize first element.
        prev_value = a_sorted[0]
        unq_first[0] = 1
        unq_first_indices[0] = 0
        count = 1

        # Loop through sorted array and identify unique values.
        for i in range(1, num_samples):
            if a_sorted[i] != prev_value:
                unq_first[i] = 1
                unq_first_indices[unq_count_idx + 1] = i
                unq_count[unq_count_idx] = count
                unq_count_idx += 1
                count = 1
                prev_value = a_sorted[i]
            else:
                unq_first[i] = 0
                count += 1

        # Assign final count.
        unq_count[unq_count_idx] = count
        unq_count_idx += 1

    # Allocate arrays for the output.
    np_unq_items = np.empty(unq_count_idx, dtype=np.int64)
    unq_idx = [None] * unq_count_idx

    for i in range(unq_count_idx):
        np_unq_items[i] = a_sorted[unq_first_indices[i]]
        unq_idx[i] = sort_idx[unq_first_indices[i]:unq_first_indices[i] + unq_count[i]]

    # Free allocated memory.
    free(unq_count)
    free(unq_first)
    free(unq_first_indices)

    return np_unq_items, unq_idx


cpdef map_indices_to_leaves(
    cnp.ndarray[intp_t, ndim=3] y_train_leaves_slice,
    cnp.ndarray[intp_t, ndim=2] bootstrap_indices,
    vector[intp_t] leaf_indices,
    vector[vector[intp_t]] leaf_values_list,
):
    """Return a mapping of training sample indices to a tree's leaf nodes.

    Parameters
    ----------
    y_train_leaves_slice : array-like of shape (n_leaves, n_outputs, n_indices)
        Unpopulated mapping representing a list of nodes, each with a list of
        indices of the training samples residing at that node.

    bootstrap_indices : array-like of shape (n_samples, n_outputs)
        Bootstrap indices of training samples.

    leaf_indices : list of ints
        List of leaf node indices. Values correspond to `leaf_values_list`.

    leaf_values_list : list of list of ints
        List of leaf node sample indices. Values correspond to `leaf_indices`.

    Returns
    -------
    y_train_leaves_slice : array-like of shape (n_leaves, n_outputs, n_indices)
        Populated mapping of training sample indices to leaf nodes. Nodes with
        no samples (e.g., internal nodes) are empty. Internal nodes are
        included so that leaf node indices match their ``est.apply`` outputs.
        Each node list is padded to equal length with 0s.
    """
    cdef intp_t n_samples, n_outputs, n_leaves
    cdef intp_t i, j, k
    cdef vector[intp_t] leaf_values
    cdef intp_t leaf_index, leaf_value, y_index
    cdef intp_t[:, :, :] y_train_leaves_slice_view

    n_outputs = bootstrap_indices.shape[1]
    n_leaves = leaf_indices.size()

    y_train_leaves_slice_view = y_train_leaves_slice  # memoryview

    with nogil:
        for i in range(n_leaves):
            leaf_index = leaf_indices[i]
            leaf_values = leaf_values_list[i]

            n_samples = leaf_values.size()
            for j in range(n_samples):
                leaf_value = leaf_values[j]
                for k in range(n_outputs):
                    y_index = bootstrap_indices[leaf_value, k]
                    if y_index > 0:
                        y_train_leaves_slice_view[leaf_index, k, j] = y_index

    return np.asarray(y_train_leaves_slice_view)
