# Skończone

from functools import partial

def curry(fun):
    return lambda x: lambda y: fun(x, y)

def curry(fun):
    return partial(partial, fun)

def add(x, y):
    return x + y

def mul(x, y):
    return x * y

print("Dodawanie przez funkcje = ", add(3, 5))
print("Dodawanie przez curry   = ", curry(add)(3)(5))
print("Mnożenie przez funkcje  = ", mul(3, 5))
print("Mnożenie przez curry    = ", curry(mul)(3)(5))

x = 5
inc = curry(add)(1)
print(f"\nTestowa wartość ({x}) + 1 = ",inc(x))