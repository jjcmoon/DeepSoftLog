from .builtins import *
from .external import External

ALL_BUILTINS = (
    External("is", 2, builtin_is),
    External("==", 2, builtin_eq),
    External("\\==", 2, builtin_neq),
    External("writeln", 1, builtin_writeln),
    External("fresh", 1, builtin_fresh),
)
