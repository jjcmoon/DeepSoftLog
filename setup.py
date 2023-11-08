from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

ext = [
    Extension("deepsoftlog.algebraic_prover.terms.expression", ["deepsoftlog/algebraic_prover/terms/expression.pyx"]),
    Extension("deepsoftlog.algebraic_prover.terms.variable", ["deepsoftlog/algebraic_prover/terms/variable.pyx"]),
    Extension(
        "deepsoftlog.algebraic_prover.terms.probability_annotation",
        ["deepsoftlog/algebraic_prover/terms/probability_annotation.pyx"],
    ),
    Extension(
        "deepsoftlog.algebraic_prover.proof_queue", ["deepsoftlog/algebraic_prover/proving/proof_queue.pyx"]
    ),
]

setup(
    name="deepsoftlog",
    author="Jaron Maene",
    author_email="jaron.maene@kuleuven.be",
    ext_modules=cythonize(ext, language_level=3),
)
