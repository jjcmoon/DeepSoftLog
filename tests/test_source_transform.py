from unittest import TestCase

from deepsoftlog.algebraic_prover.terms.expression import Fact, Expr
from deepsoftlog.logic.soft_term import BatchSoftTerm
from deepsoftlog.logic.source_transformation import SPL2ProbLog, batch_soft_rules
from deepsoftlog.parser.parser import SoftProbLogParser


class Test(TestCase):
    def test_soft_term_in_head(self):
        code = r"""
        0.5::f(~a).
        """
        code_after = r"""
        0.5::f(~SV_1) :- k(~a, ~SV_1).
        """
        self._assert_programs_same(code, code_after)

    def test_soft_term_in_head_and_body(self):
        code = r"""
        f(~a) :- g(~a), \+a.
        """
        code_after = r"""
        f(~SV_1) :- k(~a, ~SV_1), g(~a), \+a.
        """
        self._assert_programs_same(code, code_after)

    def test_soft_term_in_body(self):
        code = r"""
        f(a) :- g(~a), \+a.
        """
        code_after = r"""
        f(a) :- g(~a), \+a.
        """
        self._assert_programs_same(code, code_after)

    def test_single_var_in_head(self):
        code = r"""
        q(X) :- f(~r(X)).
        """
        code_after = r"""
        q(X) :- f(~r(X)).
        """
        self._assert_programs_same(code, code_after)

    def test_double_var_in_head(self):
        code = r"""
        q(X, r(X)) :- f(~r(X)).
        """
        code_after = r"""
        q(X, r(DV_1)) :- k(X, DV_1), f(~r(X)).
        """
        self._assert_programs_same(code, code_after)

    def _assert_programs_same(self, code1, code2):
        clauses1 = SoftProbLogParser().parse_clauses(code1)
        clauses1 = SPL2ProbLog(clauses1)
        clauses2 = SoftProbLogParser().parse_clauses(code2)
        for c1, c2 in zip(clauses1, clauses2):
            self.assertEqual(c1, c2)

    def test_soft_batching(self):
        code = """
        f(~a, g(~b)).
        f(~c, g(~d)).
        """

        clauses = SoftProbLogParser().parse_clauses(code)
        clause = batch_soft_rules(clauses).pop()
        target_clause = Fact(
            Expr("f",
                 BatchSoftTerm((Expr("a"), Expr("c"))),
                 Expr("g", BatchSoftTerm((Expr("b"), Expr("d"))))
            ))
        self.assertEqual(clause, target_clause)
