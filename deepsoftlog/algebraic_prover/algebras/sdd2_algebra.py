from typing import Union

from deepsoftlog.algebraic_prover.algebras.sdd_algebra import SddAlgebra, SddFormula
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import CompoundAlgebra, Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Fact, Expr


class ConjoinedFacts:
    def __init__(
        self, pos_facts: set[Expr], neg_facts: set[Expr], sdd_algebra: SddAlgebra
    ):
        self.pos_facts = pos_facts
        self.neg_facts = neg_facts
        self._sdd_algebra = sdd_algebra

    def __and__(self, other: "ConjoinedFacts") -> Union["ConjoinedFacts", SddFormula]:
        new_pos_facts = self.pos_facts | other.pos_facts
        new_neg_facts = self.neg_facts | other.neg_facts
        if len(new_pos_facts & new_neg_facts) != 0:
            return self._sdd_algebra.zero()
        return ConjoinedFacts(new_pos_facts, new_neg_facts, self._sdd_algebra)

    def evaluate(self, algebra: Algebra):
        return algebra.reduce_mul_value(self.pos_facts, self.neg_facts)

    def __str__(self):
        return f"ConjoinedFacts({self.pos_facts}, {self.neg_facts})"

    def __repr__(self):
        return f"ConjoinedFacts({self.pos_facts}, {self.neg_facts})"


class DnfAlgebra(CompoundAlgebra[Union[ConjoinedFacts, SddFormula]]):
    """
    Like the Sdd Algebra, but uses sets for simple conjunctions.
    This can be considerably faster, especially for programs without
    negation on rules. (in which case the knowledge compilation
    is only performed after all proofs are found).
    """

    def __init__(self, eval_algebra: Algebra):
        super().__init__(eval_algebra)
        self._sdd_algebra = SddAlgebra(eval_algebra)

    def value_pos(self, fact: Fact) -> Value:
        return self._as_conjoined_facts(pos_facts={fact})

    def value_neg(self, fact: Fact) -> Value:
        return self._as_conjoined_facts(neg_facts={fact})

    def multiply(self, v1: Value, v2: Value) -> Value:
        if not isinstance(v1, ConjoinedFacts) or not isinstance(v2, ConjoinedFacts):
            v1 = self._as_sdd(v1)
            v2 = self._as_sdd(v2)
        return v1 & v2

    def reduce_mul_value_pos(self, facts) -> Value:
        facts = {f for f in facts if f.is_annotated()}
        return self._as_conjoined_facts(facts)

    def add(self, value1: Value, value2: Value) -> Value:
        return self._as_sdd(value1) | self._as_sdd(value2)

    def one(self) -> Value:
        return self._as_conjoined_facts()

    def zero(self) -> Value:
        return self._sdd_algebra.zero()

    def reset(self):
        self._sdd_algebra.reset()

    def _as_sdd(self, value):
        if isinstance(value, ConjoinedFacts):
            return value.evaluate(self._sdd_algebra)
        return value

    def _as_conjoined_facts(self, pos_facts=None, neg_facts=None):
        pos_facts = pos_facts or set()
        neg_facts = neg_facts or set()
        return ConjoinedFacts(pos_facts, neg_facts, self._sdd_algebra)
