from typing import Optional, Iterable

from deepsoftlog.algebraic_prover.terms.expression import Expr
from ..logic.soft_term import TensorTerm

ILLEGAL_MODELS = {('.', 2), (',', 2), ('~', 1), (":-", 2), ('k', 2), ('is', 2), ('cut', 1)}
ILLEGAL_CONSTANTS = {'True', '[]'}


class Vocabulary:
    def __init__(self, constants: Optional[set] = None, functors: Optional[set] = None):
        self.constants = set() if constants is None else constants
        self.functors = set() if functors is None else functors

    def __add__(self, other):
        return Vocabulary(self.constants | other.constants, self.functors | other.functors)

    def add(self, term: Expr):
        if not isinstance(term, Expr) or isinstance(term, TensorTerm):
            return
        if term.get_arity() == 0:
            if term.functor not in ILLEGAL_CONSTANTS:
                self.constants.add(term.functor)
        else:
            signature = term.get_predicate()
            if signature not in ILLEGAL_MODELS:
                self.functors.add(signature)
            for arg in term.arguments:
                self.add(arg)
        return self

    def add_all(self, terms: Iterable[Expr]):
        for term in terms:
            self.add(term)
        return self

    def __repr__(self):
        return f"Vocabulary({self.constants}, {self.functors})"

    def get_constants(self):
        return sorted(self.constants)

    def get_functors(self):
        return sorted(self.functors)