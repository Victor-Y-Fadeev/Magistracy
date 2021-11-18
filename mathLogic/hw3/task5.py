from functools import reduce
from sympy import *


prefix = 's'
number = 1


def to_canonical_form(expr: Basic) -> Basic:
    '''Evaluate expression to Canonical form.
    
    Keyword arguments:
    expr -- logical expression

    '''
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
            return Not(reduce(And, args))
        elif isinstance(expr, Nor):
            return Not(reduce(Or, args))
        elif isinstance(expr, Implies):
            left = next(args)
            right = next(args)
            return Or(Not(left), right)
        elif isinstance(expr, Equivalent):
            left = next(args)
            right = next(args)
            return Or(And(left, right), And(Not(left), Not(right)))
        elif isinstance(expr, ITE):
            cond = next(args)
            stmt_true = next(args)
            stmt_false = next(args)
            return Or(And(cond, stmt_true), And(Not(cond), stmt_false))


def get_variable() -> str:
    ''' Get next generated variable.

    '''
    global prefix
    global number
    variable = '{}{}'.format(prefix, number)
    number += 1
    return variable

def CNF(phi: Basic, delta: set[Basic]) -> tuple[Basic, set[Basic]]:
    '''Tseytin transformation.
    
    Keyword arguments:
    phi   -- logical expression
    delta -- delta expressions

    '''
    if isinstance(phi, Symbol):
        return (phi, delta)
    elif isinstance(phi, Not):
        l, delta_prime = CNF(phi.args[0], delta)
        return (Not(l), delta_prime)
    elif isinstance(phi, And):
        variable = get_variable()
        l_1, delta_1 = CNF(phi.args[0], delta)
        l_2, delta_2 = CNF(reduce(And, phi.args[1:]), delta_1)
        return (variable, set().union(delta_2))


p, q, r = symbols('p, q, r')

expr_3a = Equivalent(p >> q, ~q >> ~p)
expr_3b = Equivalent(p >> (q >> r), ~r >> (~q >>~p))
expr_4 = ~(~(p & q) >> ~r)

#print(to_canonical_form(expr_3a))
#print(to_canonical_form(expr_3b))
#print(to_canonical_form(expr_4))

print(CNF(expr_4, (p,q,r)))

print(set((1, 2, 3) + (1, 2, 3)))