from libcpp.map cimport map
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_uint8 UINT8_t            # Unsigned 8 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef class QuantileForest:
    # The QuantileForest object.

    # Input/Output layout
    cdef public vector[vector[DOUBLE_t]] y_train
    cdef public SIZE_t[:, :, :, :] y_train_leaves
    cdef public bint sparse_pickle

    # Methods
    cpdef np.ndarray predict(
        self,
        vector[double] quantiles,
        SIZE_t[:, :] X_leaves,
        UINT8_t[:, :] X_indices=*,
        char* interpolation=*,
        bint weighted_quantile=*,
        bint weighted_leaves=*,
        bint aggregate_leaves_first=*,
    )

    cpdef np.ndarray quantile_ranks(
        self,
        double[:, :] y_scores,
        SIZE_t[:, :] X_leaves,
        UINT8_t[:, :] X_indices=*,
        char* kind=*,
        bint aggregate_leaves_first=*,
    )

    cpdef vector[map[SIZE_t, SIZE_t]] proximity_counts(
        self,
        SIZE_t[:, :] X_leaves,
        UINT8_t[:, :] X_indices=*,
        UINT32_t max_proximities=*,
        SIZE_t[:, :] sorter=*,
    )
