import math
from typing import Union

import numpy as np
import torch

FloatOrTensor = Union[float, torch.Tensor]


def safe_log(x: FloatOrTensor) -> FloatOrTensor:
    if torch.is_tensor(x):
        x[x <= 1e-12] = 0.
        return torch.log(x)
    return math.log(x)


def safe_exp(x: FloatOrTensor) -> FloatOrTensor:
    if torch.is_tensor(x):
        return torch.exp(x)
    return math.exp(x)


def safe_log_add(x: FloatOrTensor, y: FloatOrTensor) -> FloatOrTensor:
    if torch.is_tensor(x) or torch.is_tensor(y):
        return torch.logaddexp(torch.as_tensor(x), torch.as_tensor(y))
    return np.logaddexp(x, y)


def safe_log_negate(a: FloatOrTensor, eps=1e-7) -> FloatOrTensor:
    if torch.is_tensor(a):
        return torch.log1p(-torch.exp(a - eps))
    if a > -1e-10:
        return float("-inf")
    return math.log1p(-math.exp(a))
