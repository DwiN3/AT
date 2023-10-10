import random
from scipy.special import gamma
import functions as fun
import levy_flight as lf

function = ""

def generate_initial_solution(lower_bound, upper_bound):
    # Tutaj zaimplementuj losową generację początkowego rozwiązania
    return [random.uniform(lower_bound, upper_bound) for _ in range(10)]


def evaluate_solution(solution):
    # Tutaj zaimplementuj obliczanie wartości funkcji kryterialnej dla danego rozwiązania
    if function == "queue":
        fitness = fun.queueFun(solution)
    elif function == "rosenbrock":
        fitness = fun.rosenbrock_function(solution)
    elif function == "rastrigin":
        fitness = fun.rastrigin_function(solution)
    elif function == "schwefel":
        fitness = fun.schwefel_function(solution)
    else:
        fitness = sum(solution) # Przykładowe obliczenie wartości funkcji kryterialnej

    return fitness


def cuckoo_search_algorithm(population_size, max_iterations, lower_bound, upper_bound, probability, function_):
    global function  # Deklaracja globalnej zmiennej function
    nests = [generate_initial_solution(lower_bound, upper_bound) for _ in range(population_size)]
    best_solution = nests[0]
    function = function_

    for iteration in range(max_iterations):
        for i in range(population_size):
            current_solution = nests[i]

            # Generowanie nowego rozwiązania za pomocą lotu Lévy'ego
            step_length = lf.levy_flight(1, 1.5)  # Przykładowe wartości lambda_value = 1.5
            new_solution = [current_solution[j] + step_length * random.uniform(-1, 1) for j in
                            range(len(current_solution))]

            # Sprawdzanie, czy nowe rozwiązanie mieści się w granicach Lb i Ub
            new_solution = [min(max(new_solution[j], lower_bound), upper_bound) for j in range(len(new_solution))]

            # Ocena nowego rozwiązania
            current_fitness = evaluate_solution(current_solution)
            new_fitness = evaluate_solution(new_solution)

            # Aktualizacja gniazda, jeśli nowe rozwiązanie jest lepsze
            if new_fitness < current_fitness:
                nests[i] = new_solution

            # Porzucanie części gorszych rozwiązań i ich zastąpienie nowymi
            if random.random() < probability:
                nests[i] = generate_initial_solution(lower_bound, upper_bound)

        # Sortowanie gniazd względem wartości funkcji kryterialnej
        nests.sort(key=lambda x: evaluate_solution(x))

        # Aktualizacja najlepszego rozwiązania
        if evaluate_solution(nests[0]) < evaluate_solution(best_solution):
            best_solution = nests[0]

    return best_solution

