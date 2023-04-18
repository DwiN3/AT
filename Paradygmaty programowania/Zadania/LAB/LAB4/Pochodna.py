# Skończone
import math

def derivative(f, x, h=0.0001):
  return (f(x + h) - f(x)) / h

def quadratic_fun(x):
  return x**2

def main():
  print(f"Sin(x) w punkcie 1 = {derivative(math.sin, 1):.5f}")
  print(f"Sin(x) w punkcie 2 = {derivative(math.sin, 0):.5f}")
  print(f"Funkcji kwadratowej w punkcie 1 z przyrostem 0.00001 = {derivative(quadratic_fun, 1, h=0.00001):.5f}")
main()