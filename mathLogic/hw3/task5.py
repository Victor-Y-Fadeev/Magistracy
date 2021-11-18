from functools import reduce
from sympy import *


def to_canonical_form(expr):
    if isinstance(expr, Symbol):
        return expr
    else:
        args = map(to_canonical_form, expr.args)

        if isinstance(expr, And):
            return reduce(And, args)
        elif isinstance(expr, Or):
            return reduce(Or, args)
        elif isinstance(expr, Not):
            return Not(next(args))
        elif isinstance(expr, Xor):
            left = next(args)
            right = to_canonical_form(reduce(Xor, args))
            return And(Or(left, right), Not(And(left, right)))
        elif isinstance(expr, Nand):
            return
        elif isinstance(expr, Nor):
            return
        elif isinstance(expr, Implies):
            return
        elif isinstance(expr, Equivalent):
            return
        elif isinstance(expr, ITE):
            return



p, q, r = symbols('p, q, r')

expr_3a = Equivalent(p >> q, ~q >> ~p)
expr_3b = Equivalent(p >> (q >> r), ~r >> (~q >>~p))
expr_4 = ~(~(p & q) >> ~r)

#print(Xor(p))
#print(Xor(p, q))
#print(Xor(p, q, r))

#print(Xor(p, q, r))
expr = p ^ q ^ r
print(to_canonical_form(expr))

#print(type(p))
#print(p.args)

#print(And(map(Not, p)))


#print(type(expr_3a))
#print(type(p))

#print(to_cnf(expr_3b))
#print(to_cnf(expr_4))