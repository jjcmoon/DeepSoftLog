from typing import Union

import cython

from deepsoftlog.algebraic_prover.terms.variable cimport Variable, CanonicalVariableCounter
from deepsoftlog.algebraic_prover.terms.variable import fresh_variables

ExprOrVar = Union["Expr", "Variable"]


cdef class Expr:
    """
    Underlying object to represent constants, functors,
    atoms, literals and clauses (i.e. everything that is not a variable).
    """
    def __init__(self, str functor, *args: ExprOrVar, bint infix = False):
        self.functor = functor
        self.arguments = args
        self.infix = infix
        self.__hash = 0
        self._arity = len(self.arguments)

    cpdef tuple get_predicate(self):
        return self.functor, self._arity

    cpdef int get_arity(self):
        return len(self.arguments)

    def __float__(self):
        if len(self.arguments) > 0:
            raise ValueError(f"Trying to cast {self} to float")
        else:
            return float(self.functor)

    def __int__(self):
        if len(self.arguments) > 0:
            raise ValueError(f"Trying to cast {self} to int")
        else:
            return int(self.functor)

    def __eq__(self, other):
        if not isinstance(other, Expr):
            return False
        other = cython.cast(Expr, other)
        if self.functor != other.functor or self._arity != other._arity:
            return False
        return hash(self) == hash(other) and self.arguments == other.arguments

    def __str__(self):  # pragma: no cover
        if self.get_arity() == 0:
            return self.functor
        if self.functor == ":-":
            if len(self.arguments[1].arguments) == 0:
                return f"{self.arguments[0]}."
            else:
                return f"{self.arguments[0]} :- {self.arguments[1]}."
        if self.infix:
            return str(self.functor).join(str(arg) for arg in self.arguments)
        else:
            args = ",".join(str(arg) for arg in self.arguments)
            return f"{self.functor}({args})"

    def __repr__(self) -> str:  # pragma: no cover
        if self.get_arity() == 0:
            return self.functor
        args = ",".join(repr(arg) for arg in self.arguments)
        if self.infix:
            return f"'{self.functor}'({args})"
        else:
            return f"{self.functor}({args})"

    def __hash__(self):  # Hash should be identical up to renaming of variables
        if self.__hash == 0:
            args = (x for x in self.arguments if not type(x) is Variable)
            self.__hash = hash((self.functor, *args))
        return self.__hash

    def canonical_variant(self) -> Expr:
        counter = CanonicalVariableCounter()
        return fresh_variables(self, lambda: counter.get_fresh_variable())[0]

    cpdef Expr apply_substitution(self, dict substitution):
        return self.apply_substitution_(substitution)[0]

    cpdef tuple apply_substitution_(self, dict substitution):
        changed = False
        if len(substitution) == 0:
            return self, False
        new_arguments = []
        for argument in self.arguments:
            new_argument = argument.apply_substitution_(substitution)
            changed |= new_argument[1]
            new_arguments.append(new_argument[0])
        if changed:
            new_term = self.with_args(new_arguments)
            return new_term, True
        else:
            return self, False

    cpdef Expr with_args(self, list arguments):
        return Expr(self.functor, *arguments, infix=self.infix)

    def is_ground(self) -> bool:
        return all(arg.is_ground() for arg in self.arguments)

    def is_or(self) -> bool:
        return self.functor == ";"

    def is_and(self) -> bool:
        return self.functor == ","

    def is_not(self) -> bool:
        return self.functor == r"\+"
    
    def is_clause(self) -> bool:
        return self.functor == ":-"

    def is_fact(self) -> bool:
        return self.is_clause() and self.arguments[1].get_predicate() == (",", 0)

    def is_annotated(self) -> bool:
        return False

    def without_annotation(self) -> Expr:
        return self

    def get_probability(self) -> float:
        return 1.

    def get_log_probability(self) -> float:
        return 0.

    def __and__(self, other):
        return Expr(",", self, other)

    def __or__(self, other):
        return Expr(";", self, other)

    def negate(self) -> Expr:
        if self.is_not():
            return self.arguments[0]
        return Negation(self)

    def all_variables(self) -> set:
        all_vars = (arg.all_variables() for arg in self.arguments)
        return set().union(*all_vars)

    def __lt__(self, other):
        if self.get_predicate() != other.get_predicate():
            return self.get_predicate() < other.get_predicate()
        return self.arguments < other.arguments


cpdef Expr Constant(functor):
    return Expr(str(functor))

cpdef Expr TrueTerm():
    return Constant("True")

cpdef Expr FalseTerm():
    return Constant("False")


TRUE_TERM = TrueTerm()
FALSE_TERM = FalseTerm()


cpdef Expr Negation(Expr expr):
    return Expr(r"\+", expr)


cpdef Expr Clause(Expr head, tuple body):
    return Expr(":-", head, Expr(",", *body, infix=True), infix=True)


cpdef Expr Fact(Expr fact):
    return Expr(":-", fact, Expr(",", infix=True), infix=True)
