# transition(StartState, NextState, Symbol).
transition(~0, ~ns1, ~w1).
transition(~0, ~ns2, ~w2).
transition(~1, ~ns3, ~w3).
transition(~1, ~ns4, ~w4).

run(~0, []).
run(State, [Symbol|OldString]) :-
    run(OldState, OldString),
    transition(OldState, State, Symbol).

accepts(X) :- run(~0, X).
