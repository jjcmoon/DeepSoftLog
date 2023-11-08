import builtins
import re

COLORS = [f"\033[3{i}m" for i in range(1, 9)]
COLOR_CYCLE = 0


def add_color(s: str, index: int) -> str:
    return COLORS[index % len(COLORS)] + s + "\033[0m"


def get_color() -> str:
    global COLOR_CYCLE
    COLOR_CYCLE = (COLOR_CYCLE + 1) % len(COLORS)
    return COLORS[COLOR_CYCLE]


def add_rainbow_brackets(s: str) -> str:
    depth = 0
    result = []
    for chunk in re.split(r"([\(\)])", s):
        if chunk == "(":
            result.append(add_color(chunk, depth))
            depth += 1
        elif chunk == ")":
            depth -= 1
            if depth < 0:
                return s
            result.append(add_color(chunk, depth))
        else:
            result.append(chunk)

    if depth != 0:
        return s
    return "".join(result)


def print(*args, rainbow_print: bool = False, **kwargs):
    if rainbow_print:
        args = (add_rainbow_brackets(str(a)) for a in args)
    builtins.print(*args, **kwargs)


if __name__ == "__main__":
    some_lisp = "(defun factorial (n) (if (zerop n) 1 (* n (factorial (1- n)))))"
    print(some_lisp, rainbow_print=True)
