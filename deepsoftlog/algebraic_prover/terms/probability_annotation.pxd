from deepsoftlog.algebraic_prover.terms.expression cimport Expr

cdef class ProbabilisticExpr(Expr):
    cdef readonly _prob


cdef class LogProbabilisticExpr(ProbabilisticExpr):
    pass