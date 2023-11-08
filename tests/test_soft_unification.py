import math
import unittest

from deepsoftlog.algebraic_prover.terms.expression import Expr
from deepsoftlog.parser.parser import SOFTPROBLOG_PARSER


def check_program_output(test, code, target):
    program = SOFTPROBLOG_PARSER.parse("k(X, X).\n query :- " + code, embedding_metric="dummy")
    result = program(Expr("query"))[0]
    print("RESULT", result)
    test.assertAlmostEqual(target, math.exp(result))


class TestSoftUnification(unittest.TestCase):
    def check_program_output(self, code, prob):
        check_program_output(self, code, prob)

    def test_regular_unification(self):
        self.check_program_output("k(a, a).", True)
        self.check_program_output("k(a, b).", False)
        self.check_program_output("k(X, a).", True)
        self.check_program_output("k(a, X).", True)
        self.check_program_output("k(X, Y).", True)
        self.check_program_output("k(f(a), f(a)).", True)
        self.check_program_output("k(f(a), f(b)).", False)
        self.check_program_output("k(f(a), f(X)).", True)
        # TODO: no occurs check yet
        # self.check_program_output("k(f(X), X).", False)
        self.check_program_output("k(f(a, g(X)), f(Y, g(c))).", True)

    def test_soft_unification(self):
        self.check_program_output("k(~a, ~a).", 1)
        self.check_program_output("k(~a, ~b).", 0.6)
        self.check_program_output("k(~a, X).", 1)
        self.check_program_output("k(~a, ~X).", 1)
        self.check_program_output("k(~a, ~f(b)).", 0.6)
        self.check_program_output("k(~f(a), ~f(b)).", 0.6)

        self.check_program_output("k(~a, a).", 0)
        self.check_program_output("k(~f(X), f(a)).", 0)
        self.check_program_output("k(~f(X), f(X)).", 0)
