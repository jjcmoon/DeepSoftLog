from typing import TYPE_CHECKING, Iterable, Optional

from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Fact, Expr

if TYPE_CHECKING:  # pragma: no cover
    from .proof_tree import ProofTree


class Proof:
    def __init__(
        self,
        query: Expr,
        goals: Optional[tuple[Expr, ...]] = None,
        depth: int = 0,
        proof_tree: "ProofTree" = None,
        value: Value = None,
    ):
        if goals is None:
            goals = (query,)
        self.query = query
        self.depth = depth
        self.goals: tuple[Expr, ...] = goals
        self.value = value
        self.proof_tree = proof_tree
        # print(" " * depth, "new proof", self)

    def is_complete(self) -> bool:
        return len(self.goals) == 0

    def nb_goals(self) -> int:
        return len(self.goals)

    def get_algebra(self) -> Algebra:
        return self.proof_tree.algebra

    def get_children(self) -> Iterable["Proof"]:
        if self.goals[0].functor == "\\+":
            yield from self.negation_node()
        else:
            yield from self.apply_clauses()

    def negation_node(self):
        negated_goal = self.goals[0].arguments[0]
        matches = self.proof_tree.program.all_matches(negated_goal)
        if not any(matches):
            # goal is not present, so negation is trivially true
            yield self.get_child(new_goals=self.goals[1:])
        else:
            # create proof tree for negation
            sub_call_tree = self.proof_tree.get_sub_call_tree(negated_goal, self.depth)
            if sub_call_tree is None:
                yield self
            else:
                sub_call_value = sub_call_tree.value
                new_value = self.get_algebra().multiply(self.value, sub_call_value)
                yield self.get_child(new_goals=self.goals[1:], value=new_value)

    def apply_clauses(self):
        first_goal, *remaining = self.goals
        matches = self.proof_tree.program.all_matches(first_goal)
        for clause, unifier, new_facts in matches:
            new_goals = clause.arguments[1].arguments  # new goals from clause body
            new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
            query: Expr = self.query.apply_substitution(unifier)
            new_value = self.create_new_value(clause, new_facts)
            yield self.get_child(
                query=query,
                new_goals=new_goals,
                depth=self.depth + 1,
                value=new_value,
            )

    def create_new_value(self, clause, new_facts):
        new_facts = self.get_algebra().reduce_mul_value_pos(new_facts)
        new_value = self.get_algebra().multiply(self.value, new_facts)
        if clause.is_annotated():
            new_value = self.get_algebra().multiply_value_pos(new_value, clause)
        return new_value

    def get_child(
        self,
        query: Optional[Expr] = None,
        new_goals: tuple[Expr, ...] = tuple(),
        depth: Optional[int] = None,
        value: Optional[Value] = None,
    ):
        return Proof(
            query=self.query if query is None else query,
            value=self.value if value is None else value,
            goals=new_goals,
            depth=self.depth if depth is None else depth,
            proof_tree=self.proof_tree,
        )

    def __repr__(self):  # pragma: no cover
        return f"{self.query}: {self.goals} - {self.value}"

    def __lt__(self, other: "Proof"):
        return len(self.goals) < len(other.goals)
