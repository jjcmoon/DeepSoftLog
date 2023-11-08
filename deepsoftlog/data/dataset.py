import random
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TextIO, Callable

from .mutator import OutputDomainMutator, _dataset_domain, OutputMaskMutator
from ..parser.vocabulary import Vocabulary
from .query import Query


class Dataset(ABC):

    def __str__(self):
        """String with first 5 entries of the database"""
        nb_rows = min(len(self), 5)
        return "\n".join(str(self[i]) for i in range(nb_rows))

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def subset(self, i: int, j: Optional[int] = None) -> "Dataset":
        """
        :param i: index i
        :param j: index j
        :return: If j is None, returns a subset with the indices [0,i],
            else returns a subset with the indices [i, j].
        """
        if i is None:
            return self
        if j is None:
            j = i
            i = 0
        return Subset(self, i, j)

    def random_subset(self, n: Optional[int]) -> "Dataset":
        if n is None or n >= len(self):
            return self
        return RandomSubset(self, n)

    def __add__(self, other: "Dataset") -> "Dataset":
        """
        :param other: The other data.
        :return: Returns a data that is the combination of self and other
        """
        return Extension(self, other)

    def __contains__(self, item):
        if hasattr(self, "inner_dataset"):
            return item in self.inner_dataset
        # checks if dataset contains a query
        for query in self:
            if query.query == item:
                return True
        return False

    def fold(self, n: int, i: int) -> Tuple["Dataset", "Dataset"]:
        """
        :param n: The number of folds to make.
        :param i: Which of the folds is the held-out set.
        :return: A tuple of the training fold and test fold datasets.
        """
        return Fold(self, n, i, False), Fold(self, n, i, True)

    def write_to_file(self, f: TextIO):
        """
        :param f: File handle to write the data.
        """
        for query in self:
            f.write(repr(query) + "\n")

    def get_vocabulary(self):
        """
        :return: The vocabulary of the dataset.
        """
        return Vocabulary().add_all((q.query for q in self))

    def randomly_mutate_output(self, p: float = 0.5, all_outputs=False):
        mutator = OutputDomainMutator(self, change_p=True, all_outputs=all_outputs)
        return MutatingDataset(self, mutator.with_p(p))

    def mutate_output(self):
        return self + self.randomly_mutate_output(1.)

    def mutate_all_output(self, domain=None):
        return self + NegativeSpaceDataset(self, domain)

    def mask_output(self):
        return MaskSingleOutputDataset(self)

    def mask_all_output(self):
        return MutatingDataset(self, OutputMaskMutator())


class Subset(Dataset):
    def __init__(self, dataset: Dataset, i: int, j: int):
        self.i = i
        self.j = min(j, len(dataset))
        self.dataset = dataset

    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def __getitem__(self, item):
        return self.dataset[item + self.i]

    def __len__(self):
        return self.j - self.i


class RandomSubset(Dataset):
    def __init__(self, dataset: Dataset, n: int):
        self.dataset = dataset
        self.n = n
        self.ix = random.sample(range(len(dataset)), n)

    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def __getitem__(self, item):
        return self.dataset[self.ix[item]]

    def __len__(self):
        return self.n


class Extension(Dataset):
    def __init__(self, a: Dataset, b: Dataset):
        self.a = a
        self.b = b

    def __len__(self):
        return len(self.a) + len(self.b)

    def __getitem__(self, item):
        if item < len(self.a):
            return self.a[item]
        else:
            return self.b[item - len(self.a)]


class Fold(Dataset):
    def __init__(self, dataset: Dataset, n, i, split):
        self.dataset = dataset
        self.n = n
        self.i = i
        if split:
            self.indices = [i]
        else:
            self.indices = [x for x in range(n) if x != i]

        self._parent_len = len(dataset)
        self._len = len(dataset) // n * len(self.indices)
        extra = len(dataset) % n
        if i < extra:
            extra = 1 if split else extra - 1
        else:
            extra = 0 if split else extra

        self._len += extra
        self.split = split

    def __len__(self):
        return self._len

    def _get_index(self, i):
        i, j = i // (len(self.indices)), i % (len(self.indices))
        return i * self.n + self.indices[j]

    def __getitem__(self, item):
        return self.dataset[self._get_index(item)]


class MutatingDataset(Dataset):
    """
    Generic data adapter that mutates an underlying data.

    Intended use cases involve generating noisy datasets as well as negative examples.
    """

    def __init__(self, inner_dataset: Dataset, mutator: Callable[[int, Query], Query]):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.mutator = mutator

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, item):
        return self.mutator(item, self.inner_dataset[item])


class StaticDataset(Dataset):
    """Dataset from a list"""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


class NegativeSpaceDataset(StaticDataset):
    def __init__(self, dataset: Dataset, domain=None):
        if domain is None:
            domain = _dataset_domain(dataset)
        negatives = set()
        all_terms = {query.query for query in dataset}
        for query in dataset:
            for ind, values in domain.items():
                for value in values:
                    query_term = query.query
                    new_args = list(query_term.arguments)
                    new_args[ind] = value
                    query_term = query_term.with_args(new_args)
                    if query_term in all_terms:
                        continue
                    neg_query = Query(query=query_term, p=0, output_ind=query.output_ind)
                    negatives.add(neg_query)

        super().__init__(tuple(negatives))


class MaskSingleOutputDataset(Dataset):
    def __init__(self, inner_dataset):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.nb_masks = len(self.inner_dataset[0].output_ind)

    def __len__(self):
        return len(self.inner_dataset) * self.nb_masks

    def __getitem__(self, item):
        inner_item = item // self.nb_masks
        mask_index = item % self.nb_masks
        query = self.inner_dataset[inner_item]
        return query.mask_single_output(mask_index)
