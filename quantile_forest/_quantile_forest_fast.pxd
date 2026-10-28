from libcpp.map cimport map
from libcpp.vector cimport vector

import numpy as np
cimport numpy as cnp

ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t
ctypedef Py_ssize_t intp_t
ctypedef double float64_t

cdef class QuantileForest:
    # The QuantileForest object.

    # Input/Output layout.
    cdef public vector[vector[float64_t]] y_train
    # Read-only views, so that read-only (e.g., memory-mapped) arrays work, and
    # C-contiguous, since each leaf's indices are read as one block of memory.
    cdef public const intp_t[:, :, :, ::1] y_train_leaves
    cdef public const float64_t[:, :, ::1] y_bound_leaves
    cdef public bint sparse_pickle

    # Methods.
    cpdef cnp.ndarray predict(
        self,
        vector[double] quantiles,
        intp_t[:, :] X_leaves,
        uint8_t[:, :] X_indices=*,
        char* interpolation=*,
        bint weighted_quantile=*,
        bint weighted_leaves=*,
        bint aggregate_leaves_first=*,
    )

    cpdef cnp.ndarray quantile_ranks(
        self,
        double[:, :] y_scores,
        intp_t[:, :] X_leaves,
        uint8_t[:, :] X_indices=*,
        char* kind=*,
        bint aggregate_leaves_first=*,
    )

    cpdef vector[map[intp_t, int]] proximity_counts(
        self,
        intp_t[:, :] X_leaves,
        uint8_t[:, :] X_indices=*,
        uint32_t max_proximities=*,
        const intp_t[:, :] sorter=*,
    )
