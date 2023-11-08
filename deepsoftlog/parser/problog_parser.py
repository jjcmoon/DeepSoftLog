import re
from functools import reduce
from typing import Iterable

from deepsoftlog.algebraic_prover.proving.proof_module import (
    BooleanProofModule,
    ProbabilisticProofModule,
)
from deepsoftlog.algebraic_prover.terms.list_term import ConsTerm
from deepsoftlog.algebraic_prover.terms.probability_annotation import (
    ProbabilisticFact,
    ProbabilisticClause,
)
from deepsoftlog.algebraic_prover.terms.expression import *
from deepsoftlog.algebraic_prover.terms.transformations import normalize_clauses
from deepsoftlog.algebraic_prover.terms.variable import Variable


def split(
    string: str, split_symbol: str, l_brackets: str = "([", r_brackets: str = "])"
) -> list[str]:
    depth = 0
    splits = [""]
    for c in string:
        if c in l_brackets:
            depth += 1
        elif c in r_brackets:
            depth -= 1
        elif c == split_symbol and depth == 0:
            splits.append("")
            continue
        splits[-1] += c
    return splits


def clean_program(program_str: str) -> str:
    # ensure there's a trailing newline (for splitting later)
    program_str = program_str + "\n"
    # remove whitespace
    program_str = re.sub(r"[ \t]+", "", program_str)
    # remove comments
    program_str = re.sub(r"%.*\n", "\n", program_str)
    # remove empty lines
    program_str = re.sub(r"\n+", "\n", program_str)
    return program_str


class PrologParser:
    def parse_clauses(self, prolog_str: str) -> Iterable[Clause]:
        prolog_str = clean_program(prolog_str)
        clauses = [c.replace("\n", "") for c in prolog_str.split(".\n") if len(c)]
        clauses = [c.split(":-") for c in clauses]
        parsed_clauses = []
        for c in clauses:
            if len(c) == 1:
                fact = self.parse_fact(c[0])
                parsed_clauses.append(fact)

            elif len(c) == 2:
                clause = self.parse_clause(c[0], split(c[1], ","))
                parsed_clauses.append(clause)

            else:
                raise SyntaxError("Cannot parse " + repr(c))
        return parsed_clauses

    def parse_termvar(self, term_str: str) -> Union[Variable, Expr]:
        if term_str[0].isupper():
            return self.parse_variable(term_str)
        return self.parse_term(term_str)

    def parse_variable(self, var_str: str) -> Variable:
        assert var_str[0].isupper()
        return Variable(var_str)

    def parse_term(self, term_str: str) -> Expr:
        if term_str[0] == "[":
            return self.parse_list(term_str)

        start_bracket = term_str.find("(")
        if start_bracket != -1:
            functor = term_str[:start_bracket]
            arguments = split(term_str[start_bracket + 1 : -1], ",")
            return Expr(
                functor, *(self.parse_termvar(argument) for argument in arguments)
            )
        else:
            return Expr(term_str)

    def parse_list(self, list_str: str) -> Expr:
        if list_str == "[]":  # empty list
            return Expr("[]")
        if list_str[-1] != "]":
            raise SyntaxError("Cannot parse `" + repr(list_str) + "` as a list")

        head_tail = split(list_str[1:-1], "|")
        if len(head_tail) == 2:  # [head|tail] list
            list_str, tail_str = head_tail
            tail = self.parse_termvar(tail_str)
        else:
            assert len(head_tail) == 1
            list_str = head_tail[0]
            tail = Expr("[]")

        # [a,b,c] list
        args = [self.parse_termvar(arg_str) for arg_str in split(list_str, ",")]
        return reduce(lambda a, b: ConsTerm(b, a), reversed(args), tail)

    def parse_fact(self, fact) -> Fact:
        return Fact(self.parse_termvar(fact))

    def parse_atom(self, atom) -> Expr:
        return self.parse_term(atom)

    def parse_literal(self, literal) -> Expr:
        if literal.startswith(r"\+"):
            return Negation(self.parse_atom(literal[2:]))
        return self.parse_atom(literal)

    def parse_clause(self, head, body) -> Clause:
        head = self.parse_atom(head)
        body = tuple(self.parse_literal(b) for b in body)
        return Clause(head, body)

    def parse(self, prolog_str: str):
        clauses = self.parse_clauses(prolog_str)
        program = BooleanProofModule(clauses)
        return program


PROLOG_PARSER = PrologParser()


class ProbLogParser(PrologParser):
    def parse_fact(self, fact) -> Fact:
        fact = fact.split("::")
        if len(fact) == 2:
            p, fact = fact
            fact = self.parse_atom(fact)
            return ProbabilisticFact(p, fact)
        else:
            return Fact(self.parse_atom(fact[0]))

    def parse_clause(self, head, body) -> Clause:
        head = head.split("::")
        if len(head) == 2:
            p, head = head
            clause = super().parse_clause(head, body)
            return ProbabilisticClause(
                Expr(p), clause.arguments[0], clause.arguments[1]
            )
        else:
            return super().parse_clause(head[0], body)

    def parse(self, prolog_str: str, **kwargs):
        clauses = self.parse_clauses(prolog_str)
        clauses = normalize_clauses(clauses)
        program = ProbabilisticProofModule(clauses, **kwargs)
        return program


PROBLOG_PARSER = ProbLogParser()
