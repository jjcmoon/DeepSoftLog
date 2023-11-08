from collections import defaultdict
from typing import Iterable

from deepsoftlog.algebraic_prover.terms.expression import Expr, Clause
from deepsoftlog.algebraic_prover.terms.probability_annotation import ProbabilisticFact


def normalize_clauses(clauses):
    # todo, more checks?
    return eliminate_probabilistic_clauses(clauses)


def eliminate_probabilistic_clauses(clauses: Iterable[Expr]) -> Iterable[Expr]:
    """
    Transforms probabilistic clauses into normal clauses,
    by adding auxiliary probabilistic facts.
    """
    auxiliary_fact_counter = 0
    for clause in clauses:
        if not clause.is_annotated() or clause.is_fact():
            yield clause
        else:
            auxiliary_fact_counter += 1
            yield from _transform_probabilistic_clause(clause, auxiliary_fact_counter)


def _transform_probabilistic_clause(clause: Expr, unique_id: int) -> Iterable[Expr]:
    clause_head = clause.arguments[0]
    clause_body = clause.arguments[1].arguments
    auxiliary_term = Expr(f"aux_pc_{unique_id}", *clause_head.all_variables())
    yield ProbabilisticFact(clause.get_probability(), auxiliary_term)
    yield Clause(clause_head, clause_body + (auxiliary_term,))
