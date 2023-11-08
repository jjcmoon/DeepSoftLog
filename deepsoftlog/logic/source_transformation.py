from collections import defaultdict

from deepsoftlog.algebraic_prover.terms.expression import Clause, Expr, And, Fact
from deepsoftlog.algebraic_prover.terms.variable import CanonicalVariableCounter, Variable
from logic.soft_term import SoftTerm, BatchSoftTerm


def batch_soft_rules(clauses):
    """
    Combines the rules that have the same head, but different soft terms
    into a single rule with a batched soft terms.
    """
    clause_sets = defaultdict(list)
    for clause in clauses:
        clause_sets[remove_soft_term(clause)].append(clause)
    clauses = {_batch_soft_set(clause_set) for clause_set in clause_sets.values()}
    for clause in clauses:
        print(clause)
    return clauses


UID = 0


def remove_soft_term(term: Expr) -> Expr:
    global UID
    """
    Returns a term, where all soft terms are replaced
    TODO: also take variables in account
    """
    if isinstance(term, Expr):
        if term.is_soft():
            return Expr("~")
        else:
            return term.with_args([remove_soft_term(arg) for arg in term.arguments])
    UID += 1
    return Expr(f"VAR_{UID}")


def _batch_soft_set(clauses: list) -> Expr:
    clause = clauses[0]
    if len(clauses) == 1:
        return clause

    if isinstance(clause, Variable):
        assert all(isinstance(clause, Variable) for clause in clauses)
        return clause

    predicate = clause.get_predicate()
    assert all(clause.get_predicate() == predicate for clause in clauses)

    if clauses[0].is_soft():
        assert all(clause.is_soft() for clause in clauses)
        soft_terms = tuple([clause.arguments[0] for clause in clauses])
        return BatchSoftTerm(soft_terms)

    new_args = []
    for i in range(clause.get_arity()):
        new_args.append(_batch_soft_set([clause.arguments[i] for clause in clauses]))
    return clauses[0].with_args(new_args)


def SPL2ProbLog(clauses):
    X = Variable("X")
    reflexive_rule = Fact(Expr('k', X, X))  # k(X, X).

    clauses = [soft_term_elimination(clause) for clause in clauses]
    clauses = [double_var_removal(clause) for clause in clauses]
    return clauses + [reflexive_rule]


def soft_term_elimination(clause: Clause) -> Clause:
    """
    Replaces all soft terms in the head of a clause
    with variables, and adds atoms in the body to enforce
    the variable-soft term relation: `k(Var, soft_term)`
    """
    var_counter = CanonicalVariableCounter(functor="SV_")
    fresh_variables = defaultdict(var_counter.get_fresh_variable)
    substitution, new_head = _soft_term_replace(clause.get_head(), fresh_variables)
    new_atoms = [Expr('k', st, SoftTerm(v)) for st, v in substitution.items()]
    new_body = And.create(new_atoms + list(clause.get_body()))
    return clause.with_args([new_head, new_body])


def double_var_removal(clause: Clause) -> Clause:
    """
    Makes sure all occurrences of a variable in the
    head of clause are unique.
    """
    # TODO: refactor, too similar to `soft_term_elimination`
    var_counter = CanonicalVariableCounter(functor="DV_")
    fresh_variables = defaultdict(var_counter.get_fresh_variable)
    substitution, new_head = _double_var_replace(clause.get_head(), fresh_variables)
    new_atoms = [Expr('k', st, v) for st, v in substitution.items()]
    new_body = And.create(new_atoms + list(clause.get_body()))
    return clause.with_args([new_head, new_body])


def _soft_term_replace(term: Expr, substitution: defaultdict):
    if isinstance(term, Expr):
        if term.is_soft():
            term = SoftTerm(substitution[term])
        else:
            substitution, term = _rec_subst(term, substitution, _soft_term_replace)

    return substitution, term


def _double_var_replace(term: Expr, substitution: defaultdict, bound_vars=None):
    if bound_vars is None:
        bound_vars = list()
    if isinstance(term, Variable):
        if term in bound_vars:
            term = substitution[term]
        else:
            bound_vars.append(term)
    else:
        arguments = []
        for argument in term.arguments:
            subst, new_arg = _double_var_replace(argument, substitution, bound_vars)
            arguments.append(new_arg)
        term = term.with_args(arguments)

    return substitution, term


def _rec_subst(term, subst, func):
    arguments = []
    for argument in term.arguments:
        subst, new_arg = func(argument, subst)
        arguments.append(new_arg)
    term = term.with_args(arguments)
    return subst, term
