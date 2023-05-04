# Nie Skończone

def curry(fn):
    return lambda x: lambda y: fn(x, y)

from functools import partial

def curry(fn):
    return partial(lambda f, x: partial(lambda f, y: fn(x, y), f), fn)

def add(x, y):
    return x + y

inc = curry(add)(1)
print(inc(5)) # 6