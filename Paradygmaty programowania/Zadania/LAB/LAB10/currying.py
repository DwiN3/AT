# Skończone

from functools import partial

def curryLambda(fun):
    return lambda x: lambda y: fun(x, y)

def curryPartial(fun):
    return partial(partial, fun)

def add(x, y):
    return x + y

def mul(x, y):
    return x * y

def inc(x):
    return curryPartial(add)(1)(x)

print("Dodawanie przez funkcje = ", add(3, 5))
print("Dodawanie przez curry   = ", curryLambda(add)(3)(5))
print("Mnożenie przez funkcje  = ", mul(3, 5))
print("Mnożenie przez curry    = ", curryPartial(mul)(3)(5))

x = 5
print(f"\nWartość ({x}) + 1 = ", inc(x))
