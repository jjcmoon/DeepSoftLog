import numpy as np
import torch
from torch import Tensor
from torch import nn

from ..embeddings.nn_models import LeNet5

SPECIAL_MODELS = {
    ("lenet5", 1): LeNet5,
}


class Initializer:
    def __init__(self, model: nn.Module, init_mode: str, ndim: int):
        self.ndim = ndim
        self.init_mode = init_mode
        self.model = model

    def __call__(self, x) -> Tensor | nn.Module:
        if isinstance(x, str):
            return self._initialize_constant(x)
        elif isinstance(x, tuple):
            return self._initialize_functor(*x)
        return None

    def _initialize_constant(self, name: str) -> Tensor:
        if len(name) == 1 and name.isdigit() and int(name) < self.ndim:
            # digits from 0 to 9 are one-hot embeddings and not trainable
            embedding = torch.zeros(self.ndim)
            embedding[int(name)] = 1
            # needs to be a parameter to not be auto-converted
            embedding = nn.Parameter(embedding, requires_grad=False)
        else:
            embedding = random_vector(self.ndim, self.init_mode)
            embedding = nn.Parameter(embedding)
        return embedding

    def _initialize_functor(self, name: str, arity: int) -> nn.Module:
        if (name, arity) in SPECIAL_MODELS:
            return SPECIAL_MODELS[(name, arity)](self.ndim)
        return self.model(arity, self.ndim)


def random_vector(vn: int, typ: str):
    v = torch.empty(1, vn)
    if typ.endswith("normal"):
        if typ == "kaiming_normal":
            std = 1 / np.sqrt(vn)
        elif typ == "glorot_normal":
            std = np.sqrt(2 / (1 + vn))
        else:
            assert typ == "normal"
            std = 1.0
        torch.nn.init.normal_(v, std=std)

    elif typ.endswith("uniform"):
        if typ == "kaiming_uniform":
            a = np.sqrt(3 / vn)
        elif typ == "glorot_uniform":
            a = np.sqrt(6 / (1 + vn))
        else:
            assert typ == "uniform"
            a = 1.0
        torch.nn.init.uniform_(v, -a, a)

    elif typ == "sphere":
        torch.nn.init.normal_(v, std=1)
        v.data /= torch.linalg.norm(v)
    elif typ == "positive_sphere":
        torch.nn.init.normal_(v, std=1)
        v.data = torch.abs(v.data)
        v.data /= torch.linalg.norm(v)

    else:
        raise NotImplementedError(f"Initialization '{typ}' unknown")
    return v[0].detach()
