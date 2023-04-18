# Skończone

class Ulamek:
    ilosc_ulamkow = 0

    def __init__(self, *args):
        if len(args) == 1:
            dziesietny = args[0]
            mianownik = 1
            while dziesietny % 1 != 0:
                dziesietny *= 10
                mianownik *= 10
            licznik = int(dziesietny)
            nwd = Ulamek.nwd(licznik, mianownik)
            self.licznik = licznik // nwd
            self.mianownik = mianownik // nwd
        elif len(args) == 2:
            self.licznik = args[0]
            self.mianownik = args[1]
            if self.mianownik < 0:
                self.licznik *= -1
                self.mianownik *= -1
            nwd = Ulamek.nwd(self.licznik, self.mianownik)
            self.licznik //= nwd
            self.mianownik //= nwd
        else:
            raise TypeError("Nieprawidłowa liczba argumentów")
        Ulamek.ilosc_ulamkow += 1

    def __str__(self):
        return f"{self.licznik}/{self.mianownik}"

    def __add__(self, other):
        if isinstance(other, Ulamek):
            mianownik = Ulamek.nww(self.mianownik, other.mianownik)
            licznik = self.licznik * (mianownik // self.mianownik) + other.licznik * (mianownik // other.mianownik)
            return Ulamek.skroc(licznik, mianownik)
        else:
            raise TypeError('Niepoprawny typ')

    def __sub__(self, other):
        if isinstance(other, Ulamek):
            mianownik = Ulamek.nww(self.mianownik, other.mianownik)
            licznik = self.licznik * (mianownik // self.mianownik) - other.licznik * (mianownik // other.mianownik)
            return Ulamek.skroc(licznik, mianownik)
        else:
            raise TypeError('Niepoprawny typ')

    def __mul__(self, other):
        if isinstance(other, Ulamek):
            mianownik = self.mianownik * other.mianownik
            licznik = self.licznik * other.licznik
            return Ulamek.skroc(licznik, mianownik)
        else:
            raise TypeError('Niepoprawny typ')

    def __truediv__(self, other):
        if isinstance(other, Ulamek):
            mianownik = self.mianownik * other.licznik
            licznik = self.licznik * other.mianownik
            return Ulamek.skroc(licznik, mianownik)
        else:
            raise TypeError('Niepoprawny typ')

    @staticmethod
    def nww(a, b):
        return a * b // Ulamek.nwd(a, b)

    @staticmethod
    def nwd(a, b):
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def skroc(licznik, mianownik):
        nwd = Ulamek.nwd(licznik, mianownik)
        return Ulamek(licznik // nwd, mianownik // nwd)

def main():
    try:
        f1 = Ulamek(0.5)
        f2 = Ulamek(10, 6)
        print(f"Ulamek f1 = {f1}")
        print(f"Ulamek f2 = {f2}")
        print(f"f1 + f2 = {f1+f2}")
        print(f"f1 - f2 = {f1-f2}")
        print(f"f1 * f2 = {f1*f2}")
        print(f"f1 / f2 = {f1/f2}")
        print(f"Ilosc ulamkow: {f1.ilosc_ulamkow}")
    except ValueError as error:
        print(error)
main()