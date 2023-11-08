from typing import Optional

from deepsoftlog.algebraic_prover.proving.unify import replace_all_occurrences
from deepsoftlog.algebraic_prover.terms.variable import Variable
from deepsoftlog.algebraic_prover.terms.probability_annotation import LogProbabilisticExpr
from deepsoftlog.algebraic_prover.terms.expression import Expr


def get_unify_fact(term1: Expr, term2: Expr, store, metric: str) -> Expr:
    if term1 < term2:
        term1, term2 = term2, term1
    prob = store.soft_unify_score(term1, term2, metric)
    fact = Expr("k", term1, term2)
    return LogProbabilisticExpr(prob, ":-", fact, Expr(","), infix=True)


def is_soft(e: Expr):
    return e.get_predicate() == ("~", 1)


def look_for_rr(x) -> int:
    if isinstance(x, Variable):
        return 0
    elif isinstance(x, list) or isinstance(x, tuple):
        return sum(look_for_rr(t) for t in x)
    else:  # if isinstance(x, Expr):
        return x.functor.startswith("rr") + look_for_rr(x.arguments)


def soft_mgu(term1: Expr, term2: Expr, store, metric) -> Optional[tuple[dict, set]]:
    if look_for_rr([term1, term2]) > 1:
        return
    # No occurs check
    substitution = [(term1, term2)]
    soft_unifies = set()
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
                if is_soft(s) and is_soft(t):
                    s, t = s.arguments[0], t.arguments[0]
                    if isinstance(s, Variable):
                        substitution[i] = (s, t)
                    elif isinstance(t, Variable):
                        substitution[i] = (t, s)
                    elif s.is_ground() and t.is_ground():
                        if s != t:
                            soft_unifies.add(get_unify_fact(s, t, store, metric))
                        del substitution[i]
                    else:
                        raise Exception(f"Soft unification of non-ground terms `{s}` and `{t}` is illegal")
                    changes = True
                    break

                if s.get_predicate() != t.get_predicate():
                    # can't hard unify
                    return None
                new_substitution = list(zip(s.arguments, t.arguments))
                substitution = substitution[:i] + new_substitution + substitution[i+1:]
                changes = True
                break

    return dict(substitution), soft_unifies
