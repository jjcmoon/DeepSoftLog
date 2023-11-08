intvar(A, B, C, D).
==(X, Y).

addition(XS,YS,ZS) :-
    % writeln(ZS),
    to_int(ZS,Z),
    % writeln(Z),
    number(XS, X),
    number(YS, Y),
    ==(Z, +(X, Y)).


number([], 0).
number([H|T], +(*(N1, 10), D)) :-
    number(T, N1),
    fresh(D),
    intvar(0, 9, D, H).

to_int([], 0).
to_int([H|T], N) :-
    to_int(T, N1),
    is(N, plus(H, times(N1, 10))).