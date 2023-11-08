% MNIST (multi-)digit addition:
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

eq(X, X).

% Addition directly on the embeddings
addition(X,Y,Z) :- add_emb(X, Y, Z, ~0).

% Add two lists of embeddings (input1, input2, result, carry)
add_emb([], [], [], ~0).
add_emb([], [], [1], ~1).
add_emb([~HX|TX], [~HY|TY], [HZ|TZ], ~CARRY) :-
    digit(HZ),
    eq(~HZ, ~mod_ten_add(HX,HY,CARRY)),
    cut(HERE),
    add_emb(TX, TY, TZ, ~carry(HX, HY, CARRY)).


% for digit eval:
mnist(X, N) :- digit(N), eq(X, ~N).
