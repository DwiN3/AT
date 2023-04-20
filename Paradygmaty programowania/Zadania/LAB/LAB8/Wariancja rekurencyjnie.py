# Skończone

def mean_and_variance_reference(lista, dlugosc, suma):
    if not lista:
        srednia = suma / dlugosc
        return srednia, 0

    srednia, wariancja = mean_and_variance_reference(lista[1:], dlugosc, suma + lista[0])
    dlugosc -= 1
    delta = lista[0] - srednia
    wariancja = wariancja + (delta ** 2 / dlugosc)
    return srednia, wariancja

def main():
    lista1 = [3, 3, 3, 3]
    srednia1, wariancja1 = mean_and_variance_reference(lista1, len(lista1), 0)
    print(lista1)
    print(f"Średnia wynosi:   {srednia1:.3f}")
    print(f"Wariancja wynosi: {wariancja1:.3f}\n")

    lista2 = [5, 6, 7, 8, 9]
    srednia2, wariancja2 = mean_and_variance_reference(lista2, len(lista2), 0)
    print(lista2)
    print(f"Średnia wynosi:   {srednia2:.3f}")
    print(f"Wariancja wynosi: {wariancja2:.3f}\n")
main()