# Skończone
import math
from abc import ABC, abstractmethod

class Kształt(ABC):
    def __init__(self, nazwa):
        self._nazwa = nazwa

    @property
    def nazwa(self):
        return self._nazwa

    @abstractmethod
    def pole(self):
        pass

class Koło(Kształt):
    def __init__(self, nazwa, promień):
        super().__init__(nazwa)
        self._promień = promień

    def pole(self):
        return math.pi * self._promień ** 2

class Trójkąt(Kształt):
    def __init__(self, nazwa, bok_a, bok_b, bok_c):
        super().__init__(nazwa)
        self._bok_a = bok_a
        self._bok_b = bok_b
        self._bok_c = bok_c

    def pole(self):
        p = (self._bok_a + self._bok_b + self._bok_c) / 2
        return math.sqrt(p * (p - self._bok_a) * (p - self._bok_b) * (p - self._bok_c))

class Prostokąt(Kształt):
    def __init__(self, nazwa, bok_a, bok_b):
        super().__init__(nazwa)
        self._bok_a = bok_a
        self._bok_b = bok_b

    def pole(self):
        return self._bok_a * self._bok_b

class Kwadrat(Prostokąt):
    def __init__(self, nazwa, bok):
        super().__init__(nazwa, bok, bok)

class TrójkątRównoboczny(Trójkąt):
    def __init__(self, nazwa, bok):
        super().__init__(nazwa, bok, bok, bok)

def main():
    figury = [
        Koło("Koło", 2),
        Trójkąt("Trójkąt", 3, 4, 6),
        Prostokąt("Prostokąt", 4, 6),
        Kwadrat("Kwadrat", 5),
        TrójkątRównoboczny("Trójkąt równoboczny", 3),
    ]

    for figura in figury:
        print(f"{figura.nazwa}: pole = {figura.pole()}")
main()