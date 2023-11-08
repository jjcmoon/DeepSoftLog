from typing import Iterable, Optional, Callable

from .expression import Expr, ExprOrVar


class ConsTerm(Expr):
    def __init__(self, head: ExprOrVar, tail: Optional[ExprOrVar] = None):
        if tail is None:
            tail = Expr("[]")
        super().__init__(".", head, tail, infix=True)

    def with_args(self, arguments):
        return ConsTerm(*arguments)

    def __str__(self):
        return "[" + ",".join(str(t) for t in self.get_terms()) + "]"

    def __repr__(self):
        return "[" + ",".join(repr(t) for t in self.get_terms()) + "]"

    def get_terms(self) -> list[Expr, ...]:
        if isinstance(self.arguments[1], ConsTerm):
            return [self.arguments[0]] + self.arguments[1].get_terms()
        elif str(self.arguments[1]) == "[]":
            return [self.arguments[0]]
        else:
            return [self.arguments[0], self.arguments[1]]

    def __getitem__(self, item):
        if item == 0:
            return self.arguments[0]
        else:
            return self.arguments[1][item - 1]


def to_prolog_list(xs: Iterable, terminal: Callable = Expr):
    head, *tail = xs
    if isinstance(head, list):
        head = to_prolog_list(head, terminal=terminal)
    if not isinstance(head, Expr):
        head = terminal(head)
    if len(tail) == 0:
        tail = Expr("[]")
    else:
        tail = to_prolog_list(tail, terminal=terminal)
    return ConsTerm(head, tail)
