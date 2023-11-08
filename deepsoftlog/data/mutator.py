import random
from collections import defaultdict
from typing import Callable, Optional

from .query import Query


class NoiseMutatorDecorator:
    """
    Dataset mutator that will mutate with a certain probability
    """
    def __init__(
        self,
        p: float,
        inner_mutator: Callable[[int, Query], Query],
        seed: Optional[int] = None,
    ):
        """Constructor

        :param p: Probability with which to mutate the sample
        :param inner_mutator: Function that does actual mutation.
        :param seed: Seed for RNG
        """
        self.p = p
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**64)
        self.seed = seed
        self.inner_mutator = inner_mutator

    def __call__(self, index: int, query: Query) -> Query:
        rng = random.Random(self.seed ^ index)
        if rng.random() < self.p:
            return self.inner_mutator(index, query)
        else:
            return query


class OutputDomainMutator:
    """
    Dataset mutator that replaces output with an incorrect value
    from a fixed domain. Only constants are supported.
    """
    def __init__(self, dataset, change_p=False, all_outputs=False, seed: Optional[int] = None):
        """
        :param domain: Domain of outputs to choose from (per output index)
        :param change_p: If true, set the new query to have p = 1-original_p
        :param seed: Random seed
        """
        self.dataset = dataset
        self.domain = _dataset_domain(dataset)
        self.change_p = change_p
        self.all_outputs = all_outputs
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**64)
        self.seed = seed

    def with_p(self, p: float):
        return NoiseMutatorDecorator(p, self, seed=self.seed)

    def __call__(self, index: int, query: Query) -> Query:
        # We want a stable behaviour given a specific index
        rng = random.Random(self.seed ^ index)
        new_term = query.query
        output_ind = tuple(range(len(new_term.arguments))) \
            if self.all_outputs else query.output_ind

        i = rng.choice(output_ind)
        while new_term == query.query:
            new_args = list(query.query.arguments)
            new_args[i] = rng.choice(self.domain[i])
            new_term = query.query.with_args(new_args)

        return Query(
            query=new_term,
            p=(1.0 - query.p) if self.change_p else query.p,
            output_ind=query.output_ind,
        )


class OutputMaskMutator:
    def __call__(self, index: int, query: Query) -> Query:
        return query.mask_all_outputs()


def _dataset_domain(dataset: "Dataset") -> dict:
    domain = defaultdict(set)
    for query in dataset:
        for i, arg in enumerate(query.query.arguments):
            domain[i].add(arg)
    return {i: tuple(d) for i, d in domain.items()}
