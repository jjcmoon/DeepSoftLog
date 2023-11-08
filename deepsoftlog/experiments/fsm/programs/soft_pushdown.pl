# transition(StartState, NextState, Symbol).
transition(~0, ~ns1, ~w1).


run([], [~blank]).
run(cons(~symb1,T), cons(~read1,Stack)) :- run(T, Stack).
% run(cons(~symb2,T), cons(~read2,Stack)) :- run(T, Stack).
% run(~state2, cons(~symbol2,T), cons(~read2,Stack)) :- run(~prev_state2, T, cons(~write1,Stack)).
run(cons(~symb3,T), cons(~read3,Stack)) :- run(T, cons(~write1a,cons(~write1b, Stack))).
% run(cons(~symb4,T), cons(~read4,Stack)) :- run(T, cons(~write2a,cons(~write2b, Stack))).

accepts(P) :- run(P, cons(~blank, empt)).
