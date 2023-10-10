import numpy as np
import math
import warnings

def schwefel_function(solution):
    dimension = len(solution)
    sum_term = 0.0
    product_term = 1.0

    for i in range(dimension):
        sum_term += solution[i]
        product_term *= np.sin(np.sqrt(np.abs(solution[i])))

    fitness = 418.9829 * dimension - sum_term + product_term

    return fitness


def rastrigin_function(solution):
    dimension = len(solution)
    a = 10.0

    fitness = a * dimension
    for i in range(dimension):
        fitness += (solution[i] ** 2) - (a * np.cos(2 * np.pi * solution[i]))

    return fitness


def rosenbrock_function(solution):
    dimension = len(solution)
    fitness = 0

    for i in range(dimension - 1):
        fitness += 100 * (solution[i+1] - solution[i]**2)**2 + (solution[i] - 1)**2

    return fitness



warnings.simplefilter("ignore")

def queueFun(x):
    if len(x) == 0:
        raise ValueError("Input is empty")
    
    m = int(x[1])  # Konwersja na typ int
    N = x[0]
    
    p0, pN = 0.0, 0.0

    # setParametertsForQue(40.0, 20.0, 5.0, 1.0, 10.0)

    lambda_val = 40.0
    mi = 20.0
    r = 5.0
    c1 = 1.0
    c2 = 10.0

    rho = lambda_val / mi
    if m >= 0:
        mFactorial = math.factorial(m)
    else:
        mFactorial = math.gamma(abs(m) + 1)  # Silnia dla wartości ujemnej m
    
    sum_val = 0
    if m != 0:
        roDivM = rho / m
    else:
        roDivM = 0  # Można przypisać inną wartość w zależności od wymagań problemu
    
    if roDivM == 1:
        for k in range(0, m):
            sum_val += math.pow(rho, k) / math.factorial(k)
        p0 = 1 / (sum_val + (math.pow(rho, m) / mFactorial) * (N + 1.0))
    else:
        for k in range(0, m):
            sum_val += math.pow(rho, k) / math.factorial(k)
        if roDivM != 0:
            p0 = 1 / (sum_val + (math.pow(rho, m) / mFactorial) * ((1.0 - roDivM**(N + 1.0)) / (1.0 - roDivM)))
        else:
            p0 = 1 / (sum_val + (math.pow(rho, m) / mFactorial) * (N + 1.0))
    
    # Same as pOdmowy
    if m > 0:
        pN = (math.pow(rho, m + N) / math.pow(m, N)) * (p0 / mFactorial)
    else:
        pN = 0.0
    
    result = -(lambda_val * (1 - pN) * r - c1 * N - c2 * m)
    
    if math.isnan(result):
        raise ValueError("Result is NaN. Consider domain of the problem")
    
    return result
