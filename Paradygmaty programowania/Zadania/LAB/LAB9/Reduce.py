# Skończone

from functools import reduce

def srednia(lista):
    return reduce(lambda x, y: x + y, lista) / len(lista)

def najwiekszy(lista):
    return reduce(lambda x, y: x if x > y else y, lista)

def splaszcz(lista):
    return list(reduce(lambda x, y: x + y, lista))

def odleglosc(v1, v2):
    return reduce(lambda x, y: x + abs(y[0] - y[1]), zip(v1, v2), 0)

def wspolne_litery(lista):
    return list(reduce(lambda x, y: set(x) & set(y), lista))

def wstaw(lista, element):
    return list(reduce(lambda x, y: x + [element, y] if y > element and element not in x else x + [y], lista, []))

def main():
    lista1 = [1, 3, 3, 12, 4]
    print("\nFunkcja 1")
    print("Lista: ", lista1)
    print("Średnia = ", srednia(lista1))

    print("\nFunkcja 2")
    print("Lista: ", lista1)
    print("Największy: ", najwiekszy(lista1))

    lista2 = [[1, 2, 3], [5], [8, 9]]
    print("\nFunkcja 3")
    print("Lista: ", lista2)
    print("Spłaszczona lista: ", splaszcz(lista2))

    v1 = [3, 6, 9]
    v2 = [2, 4, 6]
    print("\nFunkcja 4")
    print("Wektor 1: ", v1)
    print("Wektor 2: ", v2)
    print("Odległość: ", odleglosc(v1, v2))

    lista3 = ['mama', 'ma', 'misia']
    print("\nFunkcja 5")
    print("Lista: ", lista3)
    print("Wspólne litery: ", wspolne_litery(lista3))

    lista4 = [1, 28, 71, 110, 1553]
    print("\nFunkcja 6")
    print("Lista: ", lista4)
    element_do_wstawienia = 99
    print(f'Lista z wstawionym "{element_do_wstawienia}":', wstaw(lista4, element_do_wstawienia))

main()