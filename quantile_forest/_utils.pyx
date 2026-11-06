from libc.stdlib cimport free, malloc
from libcpp.vector cimport vector

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef void parallel_qsort_asc(
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
    # A stable sort keeps each group's indices in their original order. An
    # unstable sort orders ties differently on different CPUs (NumPy's SIMD
    # sorts), which would change the samples that leaf subsampling keeps.
    sort_idx = np.argsort(a, kind="stable")
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
