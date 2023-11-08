from deepsoftlog.algebraic_prover.terms.expression import Constant, Expr

from .query import Query
from ..logic.soft_term import SoftTerm, TensorTerm


def load_tsv_file(filename: str):
    with open(filename, "r") as f:
        return [line.strip().split("\t") for line in f.readlines()]


def data_to_prolog(rows, name="r", **kwargs):
    for row in rows:
        args = [Constant(a) for a in row]
        args = [args[1], args[0], args[2]]
        args = [SoftTerm(a) for a in args]
        yield Query(Expr(name, *args), **kwargs)


def to_prolog_image(img):
    return SoftTerm(Expr("lenet5", TensorTerm(img)))
