import torch
from torch import Tensor

from deepsoftlog.algebraic_prover.algebras import safe_log


def embedding_similarity(x: Tensor, y: Tensor, distance_metric: str):
    """Returns a negative distance of two vectors in [0, 1]"""
    distance = -_get_distance(x, y, distance_metric)
    assert not torch.isnan(distance), f"Distance {distance_metric} is NaN: {x}, {y}"
    return distance


def _get_distance(x: Tensor, y: Tensor, distance_metric: str):
    """
    Returns a distance between two vectors,
    according to the chosen metric
    """
    x, y = x.squeeze(), y.squeeze()
    if distance_metric == "l1":
        # aka Manhattan distance
        d = torch.linalg.norm(x - y, ord=1)
    elif distance_metric == "l2":
        # aka Euclidean distance
        d = torch.linalg.norm(x - y)
    elif distance_metric == "gaussian":
        # l2 squared (not a distance!)
        d = torch.linalg.norm(x - y) ** 2
    elif distance_metric == "l2_normalised":
        d = normalised_l2_distance(x, y)
    elif distance_metric == "angle":
        # arccos of cosine similarity
        d = angular_distance(x, y)
    elif distance_metric == "dot":
        # dot product (in log space)
        x, y = x.abs(), y.abs()
        x = x / x.sum()
        y = y / y.sum()
        d = torch.dot(x, y).clamp(max=1.)
        d = -safe_log(d)
    else:
        raise ValueError(f"Unknown metric `{distance_metric}`")
    return d


def normalised_l2_distance(x: Tensor, y: Tensor, scale: float = 2):
    x, y = normalize(x), normalize(y)
    return torch.linalg.norm(x - y) * scale


def angular_distance(x: Tensor, y: Tensor, scale: float = 4, eps=1e-6):
    cosine_similarity = torch.nn.functional.cosine_similarity(x, y, 0)
    # careful: d\dx exp^(-arccos(x)) approaches infinity for x=1
    cosine_similarity -= eps
    return torch.arccos(cosine_similarity) * scale


def normalize(v):
    return torch.nn.functional.normalize(v, p=2, dim=0)
