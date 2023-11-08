digit(0).
digit(1).
digit(2).
digit(3).
digit(4).
digit(5).
digit(6).
digit(7).
digit(8).
digit(9).

addition(X,Y,Z) :- add_(X, Y, 0, Z).

add_([], [], 0, []).
add_([], [], 1, [1]).
add_([IMG1|T1], [IMG2|T2], CARRY, [D3|T3]) :-
    digit(D1),
    eq(~D1, IMG1),
    digit(D3),
    is(D2, rem(minus(D3, plus(D1, CARRY)), 10)),
    eq(~D2, IMG2),
    is(NEW_CARRY, div(plus(D1, plus(D2, CARRY)), 10)),
    % writeln([adding, D1, D2, CARRY, giving, NEW_CARRY, D3]),
    add_(T1, T2, NEW_CARRY, T3).

mnist(X, N) :- digit(N), eq(X, ~N).
eq(X, X).