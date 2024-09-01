from libcpp.vector cimport vector

ctypedef Py_ssize_t intp_t


cdef void parallel_qsort_asc(
    vector[double]& a,
    vector[double]& b,
    int left,
    int right,
) noexcept nogil
