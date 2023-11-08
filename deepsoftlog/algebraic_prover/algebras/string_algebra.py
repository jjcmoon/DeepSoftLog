from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra


class StringAlgebra(Algebra[str]):
    """
    String.
    Useful for debugging (but not an actual semiring!)
    """

    def value_pos(self, fact) -> str:
        return str(fact)

    def value_neg(self, fact) -> str:
        return f"!{fact}"

    def add(self, value1: str, value2: str) -> str:
        return f"{value1} | {value2}"

    def multiply(self, value1: str, value2: str) -> str:
        return f"({value1}&{value2})"

    def one(self) -> str:
        return "TRUE"

    def zero(self) -> str:
        return "FALSE"


STRING_ALGEBRA = StringAlgebra()
