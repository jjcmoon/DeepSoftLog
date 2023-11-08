from deepsoftlog.algebraic_prover.algebras import safe_log_add, safe_log_negate
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Fact


class ProbabilityAlgebra(Algebra[float]):
    def value_pos(self, f: Fact) -> float:
        return f.get_probability()

    def value_neg(self, f: Fact) -> float:
        if f.is_annotated():
            return 1 - f.get_probability()
        return 1.0

    def in_domain(self, v: float) -> bool:
        return 0 <= v <= 1

    def multiply(self, value1: float, value2: float) -> float:
        return value1 * value2

    def add(self, value1: float, value2: float) -> float:
        return value1 + value2

    def one(self) -> float:
        return 1.0

    def zero(self) -> float:
        return 0.0


class LogProbabilityAlgebra(Algebra[float]):
    """
    Probability algebra on facts in logspace.
    """

    ninf = float("-inf")

    def value_pos(self, f: Fact) -> Value:
        return f.get_log_probability()

    def value_neg(self, f: Fact) -> float:
        if f.is_annotated():
            return safe_log_negate(f.get_log_probability())
        return 0.0

    def in_domain(self, v: Value) -> bool:
        return v <= 1e-12

    def one(self) -> float:
        return 0.0

    def zero(self) -> float:
        return self.ninf

    def add(self, a: float, b: float) -> float:
        return safe_log_add(a, b)

    def multiply(self, a: float, b: float) -> float:
        return a + b


PROBABILITY_ALGEBRA = ProbabilityAlgebra()
LOG_PROBABILITY_ALGEBRA = LogProbabilityAlgebra()
