import math

import torch

from deepsoftlog.embeddings.distance import embedding_similarity
from unittest import TestCase


def get_all_similarities(t1, t2):
    for metric in ("l2", "angle", "l2_normalised"):
        yield math.exp(embedding_similarity(t1, t2, metric).item())


class Test(TestCase):
    def test_self_similar(self):
        t = torch.tensor([-0.1, 4, 2])
        for s in get_all_similarities(t, t):
            self.assertLess(abs(s - 1), 0.1)

    def test_far(self):
        t1 = torch.zeros(32)
        t1[0] = 1.
        t2 = torch.ones(32)
        for s in get_all_similarities(t1, t2):
            self.assertLess(s, 0.1)
