from collections import defaultdict
from typing import Iterable, Optional

from deepsoftlog.algebraic_prover.builtins import ALL_BUILTINS
from deepsoftlog.algebraic_prover.proving.proof_queue import OrderedProofQueue, ProofQueue
from deepsoftlog.algebraic_prover.proving.proof_tree import ProofTree
from deepsoftlog.algebraic_prover.proving.unify import mgu
from deepsoftlog.algebraic_prover.algebras.boolean_algebra import BOOLEAN_ALGEBRA
from deepsoftlog.algebraic_prover.algebras.sdd2_algebra import DnfAlgebra
from deepsoftlog.algebraic_prover.algebras.probability_algebra import (
    LOG_PROBABILITY_ALGEBRA,
    PROBABILITY_ALGEBRA,
)
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Clause, Expr
from deepsoftlog.algebraic_prover.terms.variable import CanonicalVariableCounter, fresh_variables


class ProofModule:
    def __init__(
        self,
        clauses: Iterable[Clause],
        algebra: Algebra,
    ):
        super().__init__()
        self.clauses: set[Clause] = set(clauses)
        self.algebra = algebra
        self.fresh_var_counter = CanonicalVariableCounter(functor="FV_")
        self.queried = None
        self.mask_query = False

    def mgu(self, t1, t2):
        return mgu(t1, t2)

    def all_matches(self, term: Expr) -> Iterable[tuple[Clause, dict]]:
        predicate = term.get_predicate()
        for builtin in self.get_builtins():
            if predicate == builtin.predicate:
                yield from builtin.get_answers(*term.arguments)

        for db_clause in self.clauses:
            db_head = db_clause.arguments[0]
            if self.mask_query and db_head == self.queried:
                continue  # mask the query itself
            if db_head.get_predicate() == predicate:
                fresh_db_clause = self.fresh_variables(db_clause)
                result = self.mgu(term, fresh_db_clause.arguments[0])
                if result is not None:
                    unifier, new_facts = result
                    new_clause = fresh_db_clause.apply_substitution(unifier)
                    yield new_clause, unifier, new_facts

    def fresh_variables(self, term: Clause) -> Clause:
        """Replace all variables in a clause with fresh variables"""
        return fresh_variables(term, self.fresh_var_counter.get_fresh_variable)[0]

    def get_builtins(self):
        return ALL_BUILTINS

    def query(
        self,
        query: Expr,
        max_proofs: Optional[int] = None,
        max_depth: Optional[int] = None,
        max_branching: Optional[int] = None,
        queue: Optional[ProofQueue] = None,
        return_stats: bool = False,
    ):
        self.queried = query
        if queue is None:
            queue = OrderedProofQueue(self.algebra)
        formulas, proof_steps, nb_proofs = get_proofs(
            self,
            self.algebra,
            query=query,
            max_proofs=max_proofs,
            max_depth=max_depth,
            queue=queue,
            max_branching=max_branching,
        )

        result = {k: self.algebra.evaluate(f) for k, f in formulas.items()}
        zero = self.algebra.eval_zero()
        result = {k: v for k, v in result.items() if v != zero}
        if return_stats:
            return result, proof_steps, nb_proofs
        return result

    def __call__(self, query: Expr, **kwargs):
        result, proof_steps, nb_proofs = self.query(query, return_stats=True, **kwargs)
        if type(result) is set:
            return len(result) > 0.0, proof_steps, nb_proofs
        if type(result) is dict and query in result:
            return result[query], proof_steps, nb_proofs
        return self.algebra.evaluate(self.algebra.zero()), proof_steps, nb_proofs

    def eval(self):
        self.store = self.store.eval()
        return self

    def apply(self, *args, **kwargs):
        return self.store.apply(*args, **kwargs)

    def modules(self):
        return self.store.modules()


class BooleanProofModule(ProofModule):
    def __init__(self, clauses):
        super().__init__(clauses, algebra=BOOLEAN_ALGEBRA)

    def query(self, *args, **kwargs):
        result = super().query(*args, **kwargs)
        return set(result.keys())


class ProbabilisticProofModule(ProofModule):
    def __init__(self, clauses, log_mode=False):
        eval_algebra = LOG_PROBABILITY_ALGEBRA if log_mode else PROBABILITY_ALGEBRA
        super().__init__(clauses, algebra=DnfAlgebra(eval_algebra))


def get_proofs(prover, algebra, **kwargs) -> tuple[dict[Expr, Value], int, int]:
    proof_tree = ProofTree(prover, algebra=algebra, **kwargs)
    proofs = defaultdict(algebra.zero)
    nb_proofs = 0
    for proof in proof_tree.get_proofs():
        proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
        nb_proofs += 1

    # print("ALL PROOFS", {answer: algebra.evaluate(proof) for answer, proof in proofs.items()})
    return dict(proofs), proof_tree.nb_steps, nb_proofs
