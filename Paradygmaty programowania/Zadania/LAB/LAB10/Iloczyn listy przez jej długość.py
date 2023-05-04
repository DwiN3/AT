# Skończone

from functools import partial

def iloczynLambda(lista):
    return list(map(lambda x: x * len(lista), lista))

def iloczynMul(lista):
    return list(map(partial(lambda x, y: x * y, len(lista)), lista))

def iloczynDomknięcia(lista):
    def pomnozPrzezDlugosc(x):
        return x * len(lista)
    return list(map(pomnozPrzezDlugosc, lista))

def main():
    list = [1, 4, 6, 7]
    print("Lambda:     ", iloczynLambda(list))
    print("Mul:        ", iloczynMul(list))
    print("Domknięcie: ", iloczynDomknięcia(list))

main()