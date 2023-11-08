from pathlib import Path
from typing import Union

from deepsoftlog.parser.problog_parser import ProbLogParser
from deepsoftlog.algebraic_prover.terms.transformations import normalize_clauses
from deepsoftlog.logic.soft_term import SoftTerm
from deepsoftlog.logic.spl_module import SoftProofModule


class SoftProbLogParser(ProbLogParser):
    def parse_termvar(self, term_str: str):
        is_soft = term_str.startswith("~")
        if is_soft:
            term_str = term_str[1:]
        term = super().parse_termvar(term_str)
        if is_soft:
            term = SoftTerm(term)
        return term

    def parse(self, prolog_str: str, **kwargs):
        clauses = self.parse_clauses(prolog_str)
        clauses = normalize_clauses(clauses)
        program = SoftProofModule(clauses, **kwargs)
        return program


SOFTPROBLOG_PARSER = SoftProbLogParser()


def parse_file(file_name: Union[str, Path], **kwargs):
    with open(file_name, 'r') as f:
        problog_str = f.read()
    return SOFTPROBLOG_PARSER.parse(problog_str, **kwargs)


def parse_files(*file_names: str):
    problog_str = ""
    for file_name in file_names:
        with open(file_name, 'r') as f:
            problog_str += f.read()
        problog_str += "\n"
    return SOFTPROBLOG_PARSER.parse(problog_str)
