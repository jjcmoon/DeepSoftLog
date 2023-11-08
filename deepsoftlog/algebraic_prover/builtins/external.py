from typing import Callable, Iterable, Tuple

from deepsoftlog.algebraic_prover.terms.expression import Clause, Fact, Expr


class External:
    def __init__(
        self,
        functor: str,
        arity: int,
        function: Callable[..., Tuple[Expr, dict]],
    ):
        self.functor = functor
        self.arity = arity
        self.predicate = (functor, arity)
        self.function = function

    def get_answers(self, *arguments) -> Iterable[tuple[Fact, dict, set]]:
        all_answers = self.function(*arguments)
        term = Expr(self.functor, *arguments)
        return (
            (Fact(term.apply_substitution(answer_sub)), answer_sub, set())
            for answer_sub in all_answers
        )
