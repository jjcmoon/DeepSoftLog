countries(~rr1, X, Y) :- countries(rr1, Y, Z).
countries(~rr2, X, Y) :- countries(~rr3, X, Z), countries(~rr3, Z, Y).
countries(~rr4, X, Y) :- countries(~rr5, X, Z), countries(~rr6, Z, Y).