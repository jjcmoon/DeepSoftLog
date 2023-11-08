% `act` models the result of a movement on the tape
% act(action_type, left_tape_in, symbol_in, right_tape_in, left_tape_out, symbol_out, right_tape_out).
act(~right,tape(L,S,cons(RH,RT)),  tape(cons(S,L),RH,RT)).
act(~right,tape(L,S,empt),  tape(cons(S,L),~b,empt)).

act(~left,tape([LH|LT],S,R),  tape(LT,LH,[S,R])).
act(~left,tape([],S,R),  tape([],~b,cons(S,R))).

act(~stay,Tape,  Tape).

step(State,Tape,  NewState,NewTape) :-
	transition_tape(State,Tape,  WrittenTape,Action,NewState),
	act(Action,WrittenTape, NewTape).

transition_tape(State, tape(L, Symbol, R),  tape(L, NewSymbol, R),Action,NewState) :-
    transition(State, Symbol, NewSymbol, Action, NewState).

% states
% transition(~s1, ~r1, ~w1, ~a1, ~o1).
% transition(~s1, ~b, ~p0, ~a2, ~s2).
transition(~s2, ~r2, ~w2, ~a2, ~o2).
transition(~s3, ~r3, ~w3, ~a3, ~o3).
transition(~s4, ~r4, ~w4, ~a4, ~o4).


run(~start_state, tape(empt, ~b, empt)).
run(State, Tape) :-
    run(OldState, OldTape),
    step(OldState, OldTape, State, Tape).

accepts(Tape) :- run(~halt_state, Tape).

% query :- act(~a4, tape(cons(~w0, empt), ~w3, empt), tape(cons(~b, cons(~p0, empt)), ~b, empt)).
