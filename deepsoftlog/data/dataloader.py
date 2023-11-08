import random
from math import ceil

from .dataset import Dataset


class DataLoader:
    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool = True, seed=None
    ):
        """
        :param dataset: The data that this loader will iterate over.
        :param batch_size: The batch size.
        :param shuffle: If true, the queries are shuffled,
            otherwise, they are returned in order.
        :param seed: Seed for random shuffle.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = len(self.dataset)
        self.shuffle = shuffle
        self.epoch = 0
        self.rng = random.Random(seed)
        self._set_iter()

    def _shuffling(self):
        if callable(self.shuffle):
            return self.shuffle(self.epoch)
        return self.shuffle

    def _set_iter(self):
        if self._shuffling():
            indices = list(range(self.length))
            self.rng.shuffle(indices)
            self.i = iter(indices)
        else:
            self.i = iter(range(self.length))

    def __next__(self):
        if self.i is None:
            self.epoch += 1
            self._set_iter()
            raise StopIteration
        batch = list()
        try:
            for i in range(self.batch_size):
                batch.append(self.dataset[next(self.i)])
            return batch
        except StopIteration:
            if len(batch) == 0:
                self.epoch += 1
                self._set_iter()
                raise StopIteration
            else:
                self.i = None
            return batch

    def __iter__(self):
        return self

    def __len__(self):
        return int(ceil(self.length / self.batch_size))

    def __repr__(self):
        return "DataLoader: " + str(self.dataset[0])
