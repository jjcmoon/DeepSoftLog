from deepsoftlog.algebraic_prover.algebras import safe_exp, safe_log
from deepsoftlog.algebraic_prover.terms.expression cimport Expr


cdef class ProbabilisticExpr(Expr):
    def __init__(self, prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prob = prob

    def get_probability(self):
        return self._prob

    def get_log_probability(self):
        return safe_log(self._prob)

    def is_annotated(self):
        return True

    cpdef ProbabilisticExpr with_args(self, list arguments):
        return ProbabilisticExpr(self._prob, self.functor, *arguments, infix=self.infix)

    def without_annotation(self) -> Expr:
        return Expr(self.functor, *self.arguments, infix=self.infix)

    def __str__(self):
        return f"{self.get_probability():.2g}::{super().__str__()}"

    def __repr__(self):
        return f"{self.get_probability():.2g}::{super().__repr__()}"

    """
    def __eq__(self, other):
        return other.is_annotated() \
            and self.get_probability() == other.get_probability() \
            and super().__eq__(other)

    def __hash__(self):
        return hash((self.get_probability(), super().__hash__()))
    """


cdef class LogProbabilisticExpr(ProbabilisticExpr):

    def get_probability(self):
        return safe_exp(self._prob)

    def get_log_probability(self):
        return self._prob

    cpdef LogProbabilisticExpr with_args(self, list arguments):
        return LogProbabilisticExpr(self._prob, self.functor, *arguments, infix=self.infix)


def ProbabilisticFact(prob, fact: Expr):
    return ProbabilisticExpr(float(prob), ":-", fact, Expr(","), infix=True)

def ProbabilisticClause(prob, head: Expr, body: Expr):
    assert body.functor == ","
    return ProbabilisticExpr(float(prob), ":-", head, body, infix=True)
