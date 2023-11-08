cdef class ProofQueue:
    cdef _queue
    cdef algebra
    cdef int n
    cpdef add(self, item, value)
    cpdef add_first(self, n, queue)

cdef class OrderedProofQueue:
    cdef list _queue
    cdef algebra
    cdef int n
    cpdef add(self, item, value)
    cpdef add_first(self, n, OrderedProofQueue queue)

