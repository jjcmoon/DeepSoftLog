from queue import LifoQueue
import heapq

import torch

from ..algebras.abstract_algebra import Algebra
from .proof import Proof


cdef class ProofQueue:
    def __cinit__(self, algebra):
        self._queue = LifoQueue()
        self.algebra = algebra
        self.n = 0

    cpdef add(self, item, value):
        if value is None:
            value = item.value
        self._queue.put((self.n, item))
        self.n += 1

    def next(self) -> Proof:
        return self._queue.get()[1]

    def empty(self) -> bool:
        return self._queue.empty()

    def new(self, algebra):
        return ProofQueue(algebra)

    cpdef add_first(self, n, queue):
        """
        Add the first n proofs from the given queue to this queue.
        """
        i = 0
        n = n or float("+inf")
        while not queue.empty() and i < n:
            v = queue.next()
            self.add(v, None)
            i += 1


cdef class OrderedProofQueue:
    def __cinit__(self, algebra: Algebra):
        """
        Proof queue with an ordering determined by a algebra.
        Note: on equal values, depth-first search is used.
        """
        self._queue = []
        self.algebra = algebra
        # to keep the sorting stable, we add an index
        self.n = 0

    def _get_value(self, value):
        with torch.inference_mode():
            return self.algebra.evaluate(value)

    cpdef add(self, item, value):
        if value is None:
            value = self._get_value(item.value)
        if value != self.algebra.eval_zero():
            heapq.heappush(self._queue, (-value, self.n, item))
            self.n += 1

    def next(self) -> Proof:
        return heapq.heappop(self._queue)[-1]

    def new(self, algebra):
        return OrderedProofQueue(algebra)

    cpdef add_first(self, n, OrderedProofQueue queue):
        i = 0
        n = n or float("+inf")
        while not queue.empty() and i < n:
            value, _, item = heapq.heappop(queue._queue)
            self.add(item, -value)
            i += 1

    def empty(self) -> bool:
        return len(self._queue) == 0
