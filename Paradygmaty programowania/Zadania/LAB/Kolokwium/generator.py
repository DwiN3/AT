slownik = {"0": "zero", "1": "jeden", "2": "dwa", '3': "trzy", "4": "cztery",
           '5': "pięć", '6': "sześć",'7': "siedem", "8": "osiem", '9': "dziewięć"}

def zmieniacz(lista):
    napis = ''.join(map(str, lista))
    for litera in napis:
            if litera.isdigit():
                yield slownik[litera]
            else:
                raise ValueError(f'To nie liczba: {litera}')

def podzielne(od, do, przez=2):
    return [liczba for liczba in range(od,do) if liczba % przez == 0]

def main():
    try:
        for element in zmieniacz(podzielne(4,10)):
            print(element)
    except ValueError as error:
        print(error)
main()
