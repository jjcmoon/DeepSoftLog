import pickle

import torch
from torch import Tensor

from deepsoftlog.algebraic_prover.terms.expression import Expr


class SoftTerm(Expr):
    def __init__(self, term):
        super().__init__("~", term)

    def __str__(self):
        return f"~{self.arguments[0]}"

    def with_args(self, arguments):
        return SoftTerm(*arguments)

    def get_soft_term(self):
        return self.arguments[0]


class BatchSoftTerm(Expr):
    def __init__(self, terms):
        super().__init__("~", *terms)

    def __str__(self):
        return f"~Batch({len(self.arguments)})"

    def __repr__(self):
        return str(self)

    def with_args(self, arguments):
        return BatchSoftTerm(arguments)

    def get_soft_term(self):
        return self.arguments

    def __getitem__(self, item):
        return SoftTerm(self.arguments[item])


# monkey-patching on Expr for convenience
# Expr.__invert__ = lambda self: SoftTerm(self)
# Expr.is_soft = lambda self: self.functor == "~"


class TensorTerm(Expr):
    def __init__(self, tensor: Tensor):
        super().__init__(f"tensor{tuple(tensor.shape)}")
        self.tensor = tensor

    def get_tensor(self):
        return self.tensor

    def with_args(self, arguments):
        assert len(arguments) == 0
        return self

    def __repr__(self):
        return f"t{str(hash(self))[-3:]}"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return isinstance(other, TensorTerm) \
            and self.tensor.shape == other.tensor.shape \
            and torch.all(self.tensor == other.tensor)

    def __hash__(self):
        return hash(pickle.dumps(self.tensor))

    def show(self):
        from matplotlib import pyplot as plt
        plt.imshow(self.tensor[0], cmap='gray')
        plt.show()
