from typing import Iterable

import torch

from ..algebraic_prover.builtins import External
from ..algebraic_prover.proving.proof_module import ProofModule
from ..algebraic_prover.algebras.sdd2_algebra import DnfAlgebra
from ..algebraic_prover.algebras.probability_algebra import LOG_PROBABILITY_ALGEBRA
from ..algebraic_prover.algebras.tnorm_algebra import LogProductAlgebra, LogGodelAlgebra
from ..algebraic_prover.algebras.sdd_algebra import SddAlgebra
from ..algebraic_prover.terms.expression import Expr, Fact
from ..embeddings.embedding_store import EmbeddingStore
from ..parser.vocabulary import Vocabulary
from .soft_unify import soft_mgu


class SoftProofModule(ProofModule):
    def __init__(
            self,
            clauses: Iterable[Expr],
            embedding_metric: str = "l2",
            semantics: str = 'sdd2',
    ):
        super().__init__(clauses=clauses, algebra=None)
        self.store = EmbeddingStore(0, None, Vocabulary())
        self.builtins = super().get_builtins() + (ExternalCut(),)
        self.embedding_metric = embedding_metric
        self.semantics = semantics
        self.algebra = _get_algebra(self.semantics, self)

    def mgu(self, t1, t2):
        return soft_mgu(t1, t2, self.get_store(), self.embedding_metric)

    def query(self, *args, **kwargs):
        if self.algebra is None:
            self.algebra = _get_algebra(self.semantics, self)
        self.algebra.reset()
        return super().query(*args, **kwargs)

    def get_builtins(self):
        return self.builtins

    def get_vocabulary(self):
        return Vocabulary().add_all(self.clauses)

    def get_store(self):
        if hasattr(self.store, "module"):
            return self.store.module
        return self.store

    def parameters(self):
        yield from self.store.parameters()
        if self.semantics == "neural":
            yield from self.algebra.parameters()

    def grad_norm(self, order=2):
        grads = [p.grad.detach().data.flatten()
                 for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            return 0
        grad_norm = torch.linalg.norm(torch.hstack(grads), ord=order)
        return grad_norm


def _get_algebra(semantics, program):
    if semantics == "sdd":
        return SddAlgebra(LOG_PROBABILITY_ALGEBRA)
    elif semantics == "sdd2":
        return DnfAlgebra(LOG_PROBABILITY_ALGEBRA)
    elif semantics == "godel":
        return LogGodelAlgebra()
    elif semantics == "product":
        return LogProductAlgebra()
    raise ValueError(f"Unknown semantics: {semantics}")


class ExternalCut(External):
    def __init__(self):
        super().__init__("cut", 1, None)
        self.cache = set()

    def get_answers(self, t1) -> Iterable[tuple[Expr, dict, set]]:
        if t1 not in self.cache:
            self.cache.add(t1)
            fact = Fact(Expr("cut", t1))
            return [(fact, {}, set())]
        return []
