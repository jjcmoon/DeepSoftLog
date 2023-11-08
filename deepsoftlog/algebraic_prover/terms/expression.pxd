cdef class Expr:
    cdef readonly str functor
    cdef readonly int _arity
    cdef readonly bint infix
    cdef readonly tuple arguments
    cdef int __hash

    cpdef tuple get_predicate(self)
    cpdef int get_arity(self)
    cpdef Expr apply_substitution(self, dict substitution)
    cpdef tuple apply_substitution_(self, dict substitution)
    cpdef Expr with_args(self, list arguments)

cpdef Expr Constant(functor: object)
cpdef Expr TrueTerm()
cpdef Expr FalseTerm()
cpdef Expr Negation(Expr expr)
cpdef Expr Clause(Expr head, tuple body)
cpdef Expr Fact(Expr fact)