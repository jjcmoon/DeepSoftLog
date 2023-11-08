eq(X, X).

run(~0, []).
run(NewState, [~NewSymbol|String]) :-
    run(~State, String),
    eq(~controller(State, NewSymbol), NewState).

accepts(String) :- run(~halts, String).
