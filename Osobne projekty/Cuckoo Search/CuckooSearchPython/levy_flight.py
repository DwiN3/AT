import random
import math
import matplotlib.pyplot as plt
from scipy.special import gamma

def levy_flight(step_length, lambda_value):
    numerator = lambda_value * gamma(lambda_value) * math.sin(math.pi * lambda_value / 2)
    denominator = math.pi * ((step_length ** (1 + lambda_value)) * gamma(1 + lambda_value) * lambda_value ** 2)
    return numerator / denominator


def levy_flight_plot(step_length, lambda_value, num_steps):
    x = [0]
    y = [0]

    for _ in range(num_steps):
        U = random.normalvariate(0, 1)
        V = random.normalvariate(0, 1)
        s = (U / abs(V)) ** (1 / lambda_value)
        theta = 2 * math.pi * V

        dx = step_length * s * math.sin(theta)
        dy = step_length * s * math.cos(theta)

        x.append(x[-1] + dx)
        y.append(y[-1] + dy)

    return x, y