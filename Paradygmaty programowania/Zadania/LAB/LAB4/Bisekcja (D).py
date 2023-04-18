# Skończone

import math

def bisection_method(f, x1, x2, eps):
    if f(x1)*f(x2) > 0:
        return None
    while abs(x1-x2) > eps:
        xm = (x1 + x2) / 2
        if f(xm)*f(x1) < 0:
            x2 = xm
        else:
            x1 = xm
    return (x1 + x2) / 2
  
def polynomial(x):
    return x**3 - 2*x**2 + 4*x - 1

sin_zero = bisection_method(math.sin, -1.5, 1, 0.001)
print(f"Miejsce zerowe sinusa: {sin_zero:.5f}")

polynomial_zero = bisection_method(polynomial, -10, 10, 0.001)
print(f"Miejsce zerowe funkcji x^3-2x^2+4x-1: {polynomial_zero:.5f}")