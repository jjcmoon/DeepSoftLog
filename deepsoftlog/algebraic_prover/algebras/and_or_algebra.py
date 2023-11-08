from abc import ABC

from deepsoftlog.algebraic_prover.algebras.abstract_algebra import CompoundAlgebra
from deepsoftlog.algebraic_prover.algebras.string_algebra import STRING_ALGEBRA
from deepsoftlog.algebraic_prover.terms.color_print import get_color
from deepsoftlog.algebraic_prover.terms.expression import Fact


class AndOrFormula(ABC):
    """
    AND/OR formula (in NNF)
    """

    def __and__(self, other):
        if other is TRUE_LEAF:
            return self
        if other is FALSE_LEAF:
            return FALSE_LEAF
        if self == other:
            return self
        return AndNode(self, other)

    def __or__(self, other):
        if other is TRUE_LEAF:
            return TRUE_LEAF
        if other is FALSE_LEAF:
            return self
        if self == other:
            return self
        return OrNode(self, other)

    def evaluate(self, algebra):
        raise NotImplementedError


class AndNode(AndOrFormula):
    def __init__(self, left: AndOrFormula, right: AndOrFormula):
        self.left = left
        self.right = right

    def evaluate(self, algebra):
        return algebra.multiply(self.left, self.right)

    def __eq__(self, other):
        return (
            isinstance(other, AndNode)
            and self.left == other.left
            and self.right == other.right
        )

    def __repr__(self):
        return f"{self.left} \033[1m&\033[0m {self.right}"


class OrNode(AndOrFormula):
    def __init__(self, left: AndOrFormula, right: AndOrFormula):
        self.left = left
        self.right = right

    def evaluate(self, algebra):
        return algebra.add(self.left, self.right)

    def __eq__(self, other):
        return (
            isinstance(other, OrNode)
            and self.left == other.left
            and self.right == other.right
        )

    def __repr__(self):
        color = get_color()
        color_end = "\033[0m"
        return f"{color}({color_end}{self.left} {color}|{color_end} {self.right}{color}){color_end}"


class TrueLeaf(AndOrFormula):
    def __and__(self, other):
        return other

    def __or__(self, other):
        return self

    def evaluate(self, algebra):
        return algebra.one()

    def __eq__(self, other):
        return isinstance(other, TrueLeaf)

    def __repr__(self):
        return "true"


class FalseLeaf(AndOrFormula):
    def __and__(self, other):
        return self

    def __or__(self, other):
        return other

    def evaluate(self, algebra):
        return algebra.zero()

    def __eq__(self, other):
        return isinstance(other, FalseLeaf)

    def __repr__(self):
        return "false"


class LeafNode(AndOrFormula):
    def __init__(self, fact: Fact, negated=False):
        self.fact = fact
        self.negated = negated

    def evaluate(self, algebra):
        if self.negated:
            return algebra.value_neg(self.fact)
        else:
            return algebra.value_pos(self.fact)

    def __eq__(self, other):
        return (
            isinstance(other, LeafNode)
            and self.fact == other.fact
            and self.negated == other.negated
        )

    def __repr__(self):
        if self.negated:
            return f"!{self.fact}"
        else:
            return f"{self.fact}"


TRUE_LEAF = TrueLeaf()
FALSE_LEAF = FalseLeaf()


class AndOrAlgebra(CompoundAlgebra[AndOrFormula]):
    """
    And-or formula, with some auto-simplifications.
    (Similar to a Free Algebra)
    """

    def value_pos(self, fact: Fact) -> AndOrFormula:
        return LeafNode(fact)

    def value_neg(self, fact: Fact) -> AndOrFormula:
        return LeafNode(fact, negated=True)

    def multiply(self, value1: AndOrFormula, value2: AndOrFormula) -> AndOrFormula:
        return value1 & value2

    def add(self, value1: AndOrFormula, value2: AndOrFormula) -> AndOrFormula:
        return value1 | value2

    def one(self) -> AndOrFormula:
        return TRUE_LEAF

    def zero(self) -> AndOrFormula:
        return FALSE_LEAF


AND_OR_ALGEBRA = AndOrAlgebra(STRING_ALGEBRA)
