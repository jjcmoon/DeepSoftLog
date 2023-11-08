from deepsoftlog.algebraic_prover.algebras import safe_log_negate
from deepsoftlog.algebraic_prover.algebras.probability_algebra import (
    LogProbabilityAlgebra,
    ProbabilityAlgebra,
)


# Godel is an actual semiring, the others not.
class GodelAlgebra(ProbabilityAlgebra):
    def multiply(self, value1: float, value2: float) -> float:
        return min(value1, value2)

    def add(self, value1: float, value2: float) -> float:
        return max(value1, value2)


class LogGodelAlgebra(LogProbabilityAlgebra):
    def multiply(self, value1: float, value2: float) -> float:
        return min(value1, value2)

    def add(self, value1: float, value2: float) -> float:
        return max(value1, value2)


class ProductAlgebra(ProbabilityAlgebra):
    def add(self, value1: float, value2: float) -> float:
        return value1 + value2 - value1 * value2


class LogProductAlgebra(LogProbabilityAlgebra):
    def add(self, value1: float, value2: float) -> float:
        neg_v1 = safe_log_negate(value1)
        neg_v2 = safe_log_negate(value2)
        return safe_log_negate(neg_v1 + neg_v2)


class LukasiewiczAlgebra(ProbabilityAlgebra):
    def multiply(self, value1: float, value2: float) -> float:
        return max(0.0, value1 + value2 - 1)

    def add(self, value1: float, value2: float) -> float:
        return min(1.0, value1 + value2)


class StableProductAlgebra(ProbabilityAlgebra):
    """
    Modification of the product algebra that
    avoids vanishing/exploding gradients.
    Proposed by Badreddine, Samy, et al. "Logic tensor networks."
    """

    EPS = 1e-12

    def multiply(self, value1: float, value2: float) -> float:
        value1 = (1 - self.EPS) * value1 + self.EPS
        value2 = (1 - self.EPS) * value2 + self.EPS
        return value1 * value2

    def add(self, value1: float, value2: float) -> float:
        value1 = (1 - self.EPS) * value1
        value2 = (1 - self.EPS) * value2
        return value1 + value2 - value1 * value2
