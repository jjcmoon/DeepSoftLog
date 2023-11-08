from collections import defaultdict
from typing import Callable

from deepsoftlog.algebraic_prover.terms.expression cimport Expr


cdef class Variable:
    def __init__(self, name: str):
        self.name = name

    # noinspection PyMethodMayBeStatic
    def is_ground(self) -> bool:
        return False

    cpdef apply_substitution(self, substitution):
        return self.apply_substitution_(substitution)[0]

    cpdef tuple apply_substitution_(self, substitution):
        if self in substitution:
            return substitution[self], True
        else:
            return self, False

    def all_variables(self):
        return {self}

    def __repr__(self):
        return self.name

    def __eq__(self, other: "Variable"):
        if type(other) is not Variable:
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


cdef class CanonicalVariableCounter:
    def __init__(self, start=0, functor="VAR_"):
        self.counter = start
        self.functor = functor

    cpdef get_fresh_variable(self):
        self.counter += 1
        return Variable(self.functor + str(self.counter))




cpdef fresh_variables(Expr expr, fresh_variable_function):
    substitution = dict(fresh_variables_(expr, defaultdict(fresh_variable_function)))
    return expr.apply_substitution(substitution), substitution


cpdef fresh_variables_(Expr expr, substitution):
    for argument in expr.arguments:
        if type(argument) is Variable:
            _ = substitution[argument]
        else:
            substitution = fresh_variables_(argument, substitution)
    return substitution
