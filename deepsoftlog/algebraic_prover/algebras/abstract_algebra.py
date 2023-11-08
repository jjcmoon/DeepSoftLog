from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, Iterable, TypeVar

from deepsoftlog.algebraic_prover.terms.expression import Fact

Value = TypeVar("Value")


class Algebra(ABC, Generic[Value]):
    """Interface for a algebra on facts"""

    @abstractmethod
    def value_pos(self, fact: Fact) -> Value:
        """
        Value of a positive fact.
        Note that we assume that this is only called on annotated facts.
        """
        pass

    @abstractmethod
    def value_neg(self, fact: Fact) -> Value:
        """
        Value of a negative fact.
        Note that we assume that this is only called on annotated facts.
        """
        pass

    @abstractmethod
    def multiply(self, value1: Value, value2: Value) -> Value:
        pass

    @abstractmethod
    def add(self, value1: Value, value2: Value) -> Value:
        pass

    @abstractmethod
    def one(self) -> Value:
        pass

    @abstractmethod
    def zero(self) -> Value:
        pass

    def in_domain(self, v: Value) -> bool:
        return True

    def multiply_value_pos(self, value: Value, fact: Fact) -> Value:
        return self.multiply(value, self.value_pos(fact))

    def multiply_value_neg(self, value: Value, fact: Fact) -> Value:
        return self.multiply(value, self.value_neg(fact))

    def reduce_mul(self, values: Iterable[Value]) -> Value:
        values = tuple(values)
        if len(values) == 0:
            return self.one()
        return reduce(self.multiply, values)

    def reduce_mul_value_pos(self, facts: Iterable[Fact]) -> Value:
        values = (self.value_pos(fact) for fact in facts)
        return self.reduce_mul(values)

    def reduce_mul_value_neg(self, facts: Iterable[Fact]) -> Value:
        values = (self.value_neg(fact) for fact in facts)
        return self.reduce_mul(values)

    def reduce_mul_value(
        self, pos_facts: Iterable[Fact], neg_facts: Iterable[Fact]
    ) -> Value:
        return self.multiply(
            self.reduce_mul_value_pos(pos_facts), self.reduce_mul_value_neg(neg_facts)
        )

    def reduce_add(self, values: Iterable[Value]) -> Value:
        values = tuple(values)
        if len(values) == 0:
            return self.zero()
        return reduce(self.add, values)

    def reduce_add_value_pos(self, facts: Iterable[Fact]) -> Value:
        values = (self.value_pos(fact) for fact in facts)
        return self.reduce_add(values)

    def get_dual(self):
        return DualAlgebra(self)

    def reset(self):
        pass

    def evaluate(self, value: Value) -> Value:
        return value

    def eval_zero(self):
        return self.evaluate(self.zero())

    def eval_one(self):
        return self.evaluate(self.one())

    def is_eval_zero(self, value):
        return self.evaluate(value) == self.eval_zero()

    def is_eval_one(self, value):
        return self.evaluate(value) == self.eval_one()


class DualAlgebra(Algebra):
    def __init__(self, algebra):
        self.algebra = algebra

    def value_pos(self, fact: Fact) -> Value:
        return self.algebra.value_neg(fact)

    def value_neg(self, fact: Fact) -> Value:
        return self.algebra.value_pos(fact)

    def multiply(self, value1: Value, value2: Value) -> Value:
        return self.algebra.add(value1, value2)

    def add(self, value1: Value, value2: Value) -> Value:
        return self.algebra.multiply(value1, value2)

    def one(self) -> Value:
        return self.algebra.zero()

    def zero(self) -> Value:
        return self.algebra.one()

    def get_dual(self):
        return self.algebra

    def evaluate(self, value: Value) -> Value:
        return self.algebra.evaluate(value)


class CompoundAlgebra(Algebra[Value], ABC, Generic[Value]):
    def __init__(self, eval_algebra):
        self._eval_algebra = eval_algebra

    def evaluate(self, value):
        return value.evaluate(self._eval_algebra)
