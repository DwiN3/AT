# Skończone

from functools import partial

def iloczynLambda(lista):
    return list(map(lambda x: x * len(lista), lista))

def iloczynMul(lista):
    return list(map(partial(lambda x, y: x * y, len(lista)), lista))

def pomnozPrzezDlugosc(dlugosc):
    def funkcja(x):
        return x * dlugosc
    return funkcja

def iloczynDomknięcia(lista_liczb):
    return list(map(pomnozPrzezDlugosc(len(lista_liczb)), lista_liczb))


def main():
    list = [1, 4, 6, 7]
    print("Lambda:     ", iloczynLambda(list))
    print("Mul:        ", iloczynMul(list))
    print("Domknięcie: ", iloczynDomknięcia(list))

main()