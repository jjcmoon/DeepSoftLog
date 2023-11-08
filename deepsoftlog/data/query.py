import math
import dataclasses
from typing import Optional, List, Sequence, Iterable

from deepsoftlog.algebraic_prover.terms.expression import Expr
from deepsoftlog.algebraic_prover.terms.variable import Variable


@dataclasses.dataclass(frozen=True)
class Query:
    """
    A query for a program.

    :param query: The query term.
    :param p: The target probability of this query.
    :param output_ind: A tuple that contains the indices of the
    arguments that should be considered output arguments.
    This is relevant for testing / negative mining.
    """
    query: Expr
    p: float = 1.0
    output_ind: Sequence[int] = (2,)

    def mask_single_output(self, ix: int) -> "Query":
        """
        :return:  Returns new queries where a output
                  argument is replaced with a variable.
        """
        new_args = list(self.query.arguments)
        new_args[self.output_ind[ix]] = Variable("_V")
        return dataclasses.replace(self, query=self.query.with_args(new_args))

    def mask_all_outputs(self) -> "Query":
        """
        :return:  Returns a query where all output
                  arguments are replaced with vars.
        """
        new_args = list(self.query.arguments)
        for ix in self.output_ind:
            new_args[ix] = Variable(f"_V_{ix}")
        return dataclasses.replace(self, query=self.query.with_args(new_args))

    def replace_output(self, new_values: List[Expr]) -> "Query":
        """
        Replaces the output variables
        :param new_values: The new values in order that should replace the output variables.
        :return: The query with the out_variables replaced by the corresponding new values.
        """
        new_args = list(self.query.arguments)
        for ix, new_value in zip(self.output_ind, new_values):
            new_args[ix] = new_value
        return dataclasses.replace(self, query=self.query.with_args(new_args))

    def output_values(self) -> Iterable[Expr]:
        """
        :return: The values of the output arguments
        """
        yield from (self.query.arguments[i] for i in self.output_ind)

    def substitute(self, substitution: Optional[dict] = None) -> "Query":
        """
        :param substitution: The dictionary that will be used to perform the substitution.
        :return: A new query where the substitution is applied.
        """
        return dataclasses.replace(self, query=self.query.apply_substitution(substitution))

    def error_with(self, other: float, log_mode=True):
        if other is None:
            return 1.
        if log_mode:
            other = math.exp(other)
        return abs(self.p - float(other))

    def __repr__(self):
        """Returns the query as a probabilistic fact."""
        return f"{self.p}::{self.query}."
