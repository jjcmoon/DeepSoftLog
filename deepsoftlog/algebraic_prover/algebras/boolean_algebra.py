from .abstract_algebra import Algebra


class BooleanAlgebra(Algebra[bool]):
    """
    Boolean algebra.
    Used for boolean inference (i.e. regular prolog).
    """

    def value_pos(self, fact) -> bool:
        return fact.get_probability() > 0.0

    def value_neg(self, fact) -> bool:
        return self.value_pos(fact)

    def multiply(self, value1: bool, value2: bool) -> bool:
        return value1 and value2

    def add(self, value1: bool, value2: bool) -> bool:
        return value1 or value2

    def one(self) -> bool:
        return True

    def zero(self) -> bool:
        return False


BOOLEAN_ALGEBRA = BooleanAlgebra()
