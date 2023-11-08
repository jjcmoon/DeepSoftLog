transition(~X, ~neural_next_state(X), ~neural_next_symbol(X)).
%transition(~s3, ~ns3, ~w3).
%transition(~s4, ~ns4, ~w4).

run(~s1, empt).
run(State, cons(Symbol, OldString)) :-
    run(OldState, OldString),
    transition(OldState, State, Symbol).

accepts(X) :- run(~halts, X).

query :- accepts(empt).
