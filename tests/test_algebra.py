from typing import Iterable
from unittest import TestCase

from deepsoftlog.algebraic_prover.algebras.and_or_algebra import AND_OR_ALGEBRA
from deepsoftlog.algebraic_prover.algebras.boolean_algebra import BOOLEAN_ALGEBRA
from deepsoftlog.algebraic_prover.algebras.sdd2_algebra import DnfAlgebra
from deepsoftlog.algebraic_prover.algebras.probability_algebra import PROBABILITY_ALGEBRA, LOG_PROBABILITY_ALGEBRA
from deepsoftlog.algebraic_prover.algebras.sdd_algebra import SddAlgebra
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra
from deepsoftlog.algebraic_prover.terms.probability_annotation import ProbabilisticFact
from deepsoftlog.algebraic_prover.terms.expression import Expr, Constant


def get_semirings() -> Iterable[Algebra]:
    yield BOOLEAN_ALGEBRA
    yield PROBABILITY_ALGEBRA
    yield LOG_PROBABILITY_ALGEBRA
    yield SddAlgebra(PROBABILITY_ALGEBRA)
    yield DnfAlgebra(PROBABILITY_ALGEBRA)


def get_constants(semiring: Algebra):
    return (semiring.value_pos(ProbabilisticFact(1., Expr(x))) for x in 'abcd')


class TestTerm(TestCase):
    def checkEquality(self, a, b):
        if not (isinstance(a, bool) or isinstance(b, float)):
            a = a.evaluate(AND_OR_ALGEBRA)
            b = b.evaluate(AND_OR_ALGEBRA)
        self.assertEqual(a, b)

    def test_add(self):
        for sr in get_semirings():
            a, b, c, d = get_constants(sr)
            self.checkEquality(sr.add(a, sr.add(b, c)), sr.add(sr.add(a, b), c))
            self.checkEquality(sr.add(a, b), sr.add(b, a))
            self.checkEquality(sr.add(a, sr.zero()), a)
            self.checkEquality(sr.add(sr.zero(), a), a)

    def test_multiply(self):
        for sr in get_semirings():
            a, b, c, d = get_constants(sr)
            self.checkEquality(sr.multiply(a, sr.multiply(b, c)), sr.multiply(sr.multiply(a, b), c))
            self.checkEquality(sr.multiply(a, sr.one()), a)
            self.checkEquality(sr.multiply(sr.one(), a), a)
            self.checkEquality(sr.multiply(sr.zero(), a), sr.zero())
            self.checkEquality(sr.multiply(a, sr.zero()), sr.zero())

    def test_distributivity(self):
        for sr in get_semirings():
            a, b, c, d = get_constants(sr)

            lhs = sr.multiply(a, sr.add(b, c))
            rhs = sr.add(sr.multiply(a, b), sr.multiply(a, c))
            self.assertEqual(lhs, rhs)

            lhs = sr.multiply(sr.add(a, b), c)
            rhs = sr.add(sr.multiply(a, c), sr.multiply(b, c))
            self.assertEqual(lhs, rhs)

