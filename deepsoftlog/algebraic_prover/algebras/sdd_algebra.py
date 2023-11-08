from typing import Optional, TypeVar

from pysdd.iterator import SddIterator
from pysdd.sdd import SddManager, SddNode

from deepsoftlog.algebraic_prover.algebras.and_or_algebra import AND_OR_ALGEBRA
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra, Value, CompoundAlgebra
from deepsoftlog.algebraic_prover.algebras.string_algebra import STRING_ALGEBRA
from deepsoftlog.algebraic_prover.terms.expression import Fact

V = TypeVar("V")


class FastList:
    """
    List that allows fast access to elements by index and value.
    """

    def __init__(self):
        self._ix_to_val: list[V] = []
        self._val_to_ix: dict[V, int] = {}

    def __getitem__(self, ix: int) -> V:
        return self._ix_to_val[ix - 1]

    def __contains__(self, value: V):
        return value in self._val_to_ix

    def __len__(self) -> int:
        return len(self._ix_to_val)

    def index(self, val: V) -> int:
        return self._val_to_ix[val] + 1

    def append(self, val: V):
        self._ix_to_val.append(val)
        self._val_to_ix[val] = len(self._ix_to_val)


class SddFormula:
    """
    Self-managing SDD formula
    """

    def __init__(
        self,
        manager: SddManager,
        all_facts: FastList,
        formula: Optional[SddNode] = None,
    ):
        if formula is None:
            formula = manager.true()
        self.manager = manager
        self.all_facts = all_facts
        self.formula: SddNode = formula
        self._score: Optional[float] = None

    def _get_child(self, formula=None):
        return SddFormula(self.manager, self.all_facts, formula)

    def evaluate(self, algebra: Algebra) -> float:
        return _sdd_eval(self.manager, self.formula, algebra, self.all_facts)

    def __and__(self, other: "SddFormula") -> "SddFormula":
        return self._get_child(self.manager.conjoin(self.formula, other.formula))

    def __or__(self, other: "SddFormula") -> "SddFormula":
        return self._get_child(self.manager.disjoin(self.formula, other.formula))

    def _fact_id(self, fact: Fact, negated: bool) -> int:
        if negated:
            return -self._fact_id(fact, False)
        if fact not in self.all_facts:
            self.all_facts.append(fact)
            if len(self.all_facts) >= self.manager.var_count():
                self.manager.add_var_after_last()
        return self.all_facts.index(fact)

    def with_fact(self, fact: Fact, negated: bool = False) -> "SddFormula":
        assert fact.is_ground(), f"{fact} is not ground!"
        fact_id = self._fact_id(fact, negated)
        fact_sdd = self.manager.literal(fact_id)
        return self._get_child(self.manager.conjoin(self.formula, fact_sdd))

    def negate(self) -> "SddFormula":
        return self._get_child(self.manager.negate(self.formula))

    def view(self, save_path="temp.dot"):
        with open(save_path, "w") as out:
            print(self.formula.dot(), file=out)

        from graphviz import Source

        s = Source.from_file(save_path)
        s.view()

    def __repr__(self):
        return self.evaluate(STRING_ALGEBRA)

    def __eq__(self, other):
        return isinstance(other, SddFormula) and self.formula == other.formula

    def to_and_or_tree(self):
        return self.evaluate(AND_OR_ALGEBRA)


def _sdd_eval(
    manager: SddManager, root_node: SddNode, algebra: Algebra[Value], all_facts
) -> Value:
    iterator = SddIterator(manager, smooth=False)

    def _formula_evaluator(node: SddNode, r_values, *_):
        if node is not None:
            if node.is_literal():
                literal = node.literal
                if literal < 0:
                    return algebra.value_neg(all_facts[(-literal) - 1])
                else:
                    return algebra.value_pos(all_facts[literal - 1])
            elif node.is_true():
                return algebra.one()
            elif node.is_false():
                return algebra.zero()
        # Decision node
        return algebra.reduce_add(
            [algebra.reduce_mul(value[0:2]) for value in r_values]
        )

    result = iterator.depth_first(root_node, _formula_evaluator)
    return result


class SddAlgebra(CompoundAlgebra[SddFormula]):
    """
    SDD semiring on facts.
    Solves the disjoin-sum problem.
    """

    def __init__(self, eval_algebra):
        super().__init__(eval_algebra)
        self.manager = SddManager()
        self.all_facts = FastList()
        self._i = 0

    def value_pos(self, fact) -> SddFormula:
        return self.one().with_fact(fact)

    def value_neg(self, fact) -> SddFormula:
        return self.one().with_fact(fact).negate()

    def multiply(self, value1: SddFormula, value2: SddFormula) -> SddFormula:
        return value1 & value2

    def add(self, value1: SddFormula, value2: SddFormula) -> SddFormula:
        return value1 | value2

    def one(self) -> SddFormula:
        return SddFormula(self.manager, self.all_facts)

    def zero(self) -> SddFormula:
        return self.one().negate()

    def reset(self):
        self.all_facts = FastList()
