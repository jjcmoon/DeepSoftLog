cdef class Variable:
    cdef readonly str name

    cpdef apply_substitution(self, substitution)
    cpdef tuple apply_substitution_(self, substitution)

cdef class CanonicalVariableCounter:
    cdef int counter
    cdef str functor

    cpdef get_fresh_variable(self)