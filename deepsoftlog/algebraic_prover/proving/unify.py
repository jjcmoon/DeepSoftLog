from typing import Optional, Tuple, Union

from deepsoftlog.algebraic_prover.terms.expression import Expr
from deepsoftlog.algebraic_prover.terms.variable import Variable


def replace_occurrence(s: Variable, t: Union[Variable, Expr], x: Union[Variable, Expr]):
    if x == s:
        return t, True
    elif isinstance(x, Expr):
        new_arguments = []
        changed = False
        for argument in x.arguments:
            new_argument = replace_occurrence(s, t, argument)
            changed |= new_argument[1]
            new_arguments.append(new_argument[0])
        return x.with_args(new_arguments), changed
    return x, False


def replace_all_occurrences(
    s: Variable, t: Union[Variable, Expr], index, substitution: list
) -> Optional[list]:
    changes = False
    new_substitution = []
    for i, (lhs, rhs) in enumerate(substitution):
        if i != index:
            lhs, changed = replace_occurrence(s, t, lhs)
            changes |= changed
            rhs, changed = replace_occurrence(s, t, rhs)
            changes |= changed
        new_substitution.append((lhs, rhs))
    if not changes:
        return None
    return new_substitution


def mgu(term1: Expr, term2: Expr) -> Optional[tuple[dict, set]]:
    """
    Most General Unifier of two expressions.
    """
    # No occurs check
    substitution = [(term1, term2)]
    changes = True
    while changes:
        changes = False
        for i in range(len(substitution)):
            s, t = substitution[i]
            if type(t) is Variable and type(s) is not Variable:
                substitution[i] = (t, s)
                changes = True
                break
            if type(s) is Variable:
                if t == s:
                    del substitution[i]
                    changes = True
                    break
                new_substitution = replace_all_occurrences(s, t, i, substitution)
                if new_substitution is not None:
                    substitution = new_substitution
                    changes = True
                    break
            if isinstance(s, Expr) and isinstance(t, Expr):
                if s.get_predicate() != t.get_predicate():
                    return
                new_substitution = [
                    (s.arguments[j], t.arguments[j]) for j in range(s.get_arity())
                ]
                substitution = (
                    substitution[:i] + new_substitution + substitution[i + 1 :]
                )
                changes = True
                break

    return {k: v for (k, v) in substitution}, set()


def unify(term1: Expr, term2: Expr) -> Optional[Tuple[Expr, dict]]:
    result = mgu(term1, term2)
    if result is None:
        return None
    substitution = result[0]
    return term1.apply_substitution(substitution), substitution


def more_general_than(generic: Expr, specific: Expr) -> bool:
    # Cf. subsumes_term in SWI-Prolog
    result = mgu(generic, specific)
    if result is None:
        return False
    single_sided_unifier = result[0]
    return specific == specific.apply_substitution(single_sided_unifier)


def more_specific_than(specific: Expr, generic: Expr) -> bool:
    return more_general_than(generic, specific)
