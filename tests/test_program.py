import math
from unittest import TestCase

from deepsoftlog.algebraic_prover.terms.expression import Expr
from deepsoftlog.parser.parser import SOFTPROBLOG_PARSER


def check_program_output(test, code, prob):
    query = Expr("query")
    program = SOFTPROBLOG_PARSER.parse(code, embedding_metric='dummy', semantics='sdd2')
    result = math.exp(program(query)[0])
    test.assertAlmostEqual(prob, result)


class Test(TestCase):
    def test_simple(self):
        code = """
        0.5::f(~a).
        query :- f(~b).
        """
        check_program_output(self, code, 0.5 * 0.6)

    def test_unification(self):
        # only 1 soft-unification should be counted!
        # hence 0.6 and not 0.36
        code = """
        f(~a). 
        g(~b).
        query :- f(~b), g(~a).
        """
        check_program_output(self, code, 0.6)

    def test_contradiction(self):
        # can't both have k(a,b) and \+k(b,a)
        code = r"""
        f(~a).
        g(~b).
        query :- f(~b), \+g(~a).
        """
        check_program_output(self, code, 0.)

    def test_triangle_basic(self):
        # triangle probability is not normalized
        code = r"""
        a(~x). 
        b(~y).
        c(~z).
        query :- a(~y), b(~z), c(~x).
        """
        check_program_output(self, code, 0.6 * 0.6 * 0.6)

    def test_triangle_expansion(self):
        # naive handling of triangle relations
        code = """
        a(~x). 
        b(~y).
        query :- a(~y), b(~z).
        """
        check_program_output(self, code, 0.6 * 0.6)

    def test_triangle_contradiction(self):
        code = r"""
        a(~x). 
        b(~y).
        c(~z).
        query :- a(~y), b(~z), \+c(~x).
        """
        check_program_output(self, code, 0.6 * 0.6 * 0.4)

    def test_triangle_negative_expansion(self):
        code = r"""
        a(~x). 
        b(~y).
        c(~z).
        query :- a(~y), \+b(~z).
        """
        check_program_output(self, code, 0.6 * 0.4)

    def test_triangle_marginalize(self):
        code = r"""
        a(~x).
        b(~y).
        query :- a(~y), b(~z).
        query :- a(~y), \+b(~z).
        """
        check_program_output(self, code, 0.6)

    def test_or_combination(self):
        code = r"""
        p(~a).
        q(~a).
        0.5 :: x.
        query :- x, p(~b).
        query :- x, q(~c).
        """
        check_program_output(self, code, 0.5 * (0.6 + 0.4 * 0.6))

    def test_soft_hard_not_equal(self):
        code = r"""
        f(a).
        query :- f(~a).
        """
        check_program_output(self, code, 0.)
        code = r"""
        f(~a).
        query :- f(a).
        """
        check_program_output(self, code, 0.)
