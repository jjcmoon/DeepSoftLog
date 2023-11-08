from typing import TYPE_CHECKING, Iterator, Optional

from deepsoftlog.algebraic_prover.proving.proof import Proof
from deepsoftlog.algebraic_prover.proving.proof_queue import ProofQueue
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra
from deepsoftlog.algebraic_prover.terms.expression import Expr

if TYPE_CHECKING:
    from deepsoftlog.algebraic_prover.proving.proof_module import ProofModule


class ProofTree:
    """
    Proof tree for a query.
    Searches depth-first.
    """

    def __init__(
        self,
        program: "ProofModule",
        query: Expr,
        algebra: Algebra,
        max_depth: Optional[int] = None,
        max_proofs: Optional[int] = None,
        queue: ProofQueue = None,
        max_branching: Optional[int] = None,
    ):
        self.algebra = algebra
        self.program = program
        self.max_depth = max_depth
        self.max_proofs = max_proofs
        self.max_branching = max_branching
        self.sub_calls = dict()
        self.answers = set()
        self.incomplete_sub_trees: list["ProofTree"] = []
        self.proofs = []
        self.queue = queue
        self.queue.add(self._create_proof_for(query), None)
        self.value = self.algebra.zero()
        self.nb_steps = 0

    def _create_proof_for(self, query: Expr):
        return Proof(query=query, proof_tree=self, value=self.algebra.one())

    def is_complete(self) -> bool:
        return self.queue.empty() or len(self.proofs) >= self.get_max_proofs()

    def get_proofs(self) -> Iterator[Proof]:
        while not self.is_complete():
            proof = self.step()
            if proof is not None:
                yield proof

    def step(self) -> Optional[Proof]:
        self.nb_steps += 1
        if len(self.incomplete_sub_trees):
            return self._step_subtree()

        proof = self.queue.next()
        if proof.is_complete():
            self.answers.add(proof.query)
            self.proofs.append(proof)
            self.value = self.algebra.add(self.value, proof.value)
            return proof

        if not self.is_pruned(proof):
            local_queue = self.queue.new(self.algebra)
            proof_remaining = proof.nb_goals()
            for child_proof in proof.get_children():
                child_remaining = child_proof.nb_goals()
                if child_remaining < proof_remaining:
                    local_queue.add(child_proof, None)
                else:
                    self.queue.add(child_proof, None)
            self.queue.add_first(self.max_branching, local_queue)

    def _step_subtree(self):
        self.incomplete_sub_trees[-1].step()
        if self.incomplete_sub_trees[-1].is_complete():
            del self.incomplete_sub_trees[-1]

    def get_answers(self) -> set[Expr]:
        assert self.is_complete()
        return self.answers

    def sub_call(self, query: Expr, depth: int) -> "ProofTree":
        new_algebra = self.algebra.get_dual()
        new_tree = type(self)(
            program=self.program,
            query=query,
            algebra=new_algebra,
            max_depth=self.max_depth - depth if self.max_depth is not None else None,
            max_proofs=self.max_proofs,
            max_branching=self.max_branching,
            queue=self.queue.new(new_algebra),
        )
        self.sub_calls[query] = new_tree
        return new_tree

    def get_sub_call_tree(self, query: Expr, depth: int) -> Optional["ProofTree"]:
        if query in self.sub_calls:
            return self.sub_calls[query]
        else:
            new_tree = self.sub_call(query, depth)
            self.incomplete_sub_trees.append(new_tree)
            return None

    def get_max_depth(self):
        if self.max_depth is None:
            return float("+inf")
        return self.max_depth

    def get_max_proofs(self):
        if self.max_proofs is None:
            return float("+inf")
        return self.max_proofs

    def is_pruned(self, proof: Proof):
        if proof.depth > self.get_max_depth():
            if self.max_depth is None:  # pragma: no cover
                import warnings

                warnings.warn("Default max depth exceeded")
            return True
        return False
