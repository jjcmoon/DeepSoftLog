from deepsoftlog.algebraic_prover.terms.expression import Expr, Constant
from deepsoftlog.algebraic_prover.terms.variable import Variable

eps = 1e-6


def builtin_writeln(term: Expr):
    print(term)
    return [{}]


def eval_term(term: Expr):
    term = str(term)
    plus = lambda x, y: x + y
    times = lambda x, y: x * y
    minus = lambda x, y: x - y
    div = lambda x, y: x // y
    rem = lambda x, y: x % y
    geq = lambda x, y: x >= y
    leq = lambda x, y: x <= y
    try:
        r = eval(term)
    except Exception as e:
        print("ERROR in evaluating", term)
        raise e
    return r


def builtin_is(lhs, rhs):
    rhs = eval_term(rhs)
    if type(lhs) is Variable:
        return [{lhs: Expr(str(rhs))}]
    else:
        lhs = eval(str(lhs))
        if abs(lhs - rhs) < eps:
            return [{}]
        else:
            return []


def builtin_eq(lhs, rhs):
    if type(lhs) == type(rhs) and lhs == rhs:
        return [{}]
    else:
        return []


def builtin_neq(lhs, rhs):
    if lhs == rhs:
        return []
    else:
        return [{}]


COUNTER: int = 0


def builtin_fresh(var: Variable):
    global COUNTER
    COUNTER += 1
    return [{var: Constant(f"fresh_{COUNTER}")}]
