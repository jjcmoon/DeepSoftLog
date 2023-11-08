
start_state(0).
halt_state(0).

run(X, []) :- start_state(X).
run(State, [Symbol|OldString]) :-
    run(OldState, OldString),
    transition(OldState, State, Symbol).

accepts(X) :- halt_state(H), run(H, X).
query(accepts([])).

