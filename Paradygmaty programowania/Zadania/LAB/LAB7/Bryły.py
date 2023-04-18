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

class Mixin:
    def pole(self):
        return sum([sciana.pole() for sciana in self._sciany])

class Czworoscian(Mixin, Kształt):
    def __init__(self, nazwa, bok):
        super().__init__(nazwa)
        self._sciany = [
            TrojkatRownoboczny("sciana1", bok),
            TrojkatRownoboczny("sciana2", bok),
            TrojkatRownoboczny("sciana3", bok),
            TrojkatRownoboczny("sciana4", bok),
        ]


class Szescian(Mixin, Kształt):
    def __init__(self, nazwa, bok):
        super().__init__(nazwa)
        self._sciany = [
            Kwadrat("sciana1", bok),
            Kwadrat("sciana2", bok),
            Kwadrat("sciana3", bok),
            Kwadrat("sciana4", bok),
            Kwadrat("sciana5", bok),
            Kwadrat("sciana6", bok),
        ]


class Piramida(Mixin, Kształt):
    def __init__(self, nazwa, bok):
        super().__init__(nazwa)
        self._sciany = [
            Czworokat("sciana1", bok, bok, bok, bok),
            TrojkatRownoboczny("sciana2", bok),
            TrojkatRownoboczny("sciana3", bok),
            TrojkatRownoboczny("sciana4", bok),
        ]


class Czworokat(Kształt):
    def __init__(self, nazwa, bok_a, bok_b, bok_c, bok_d):
        super().__init__(nazwa)
        self._bok_a = bok_a
        self._bok_b = bok_b
        self._bok_c = bok_c
        self._bok_d = bok_d

    def pole(self):
        return self._bok_a ** 2


class Kwadrat(Kształt):
    def __init__(self, nazwa, bok):
        super().__init__(nazwa)
        self._bok = bok

    def pole(self):
        return self._bok ** 2

class Trójkąt(Kształt):
    def __init__(self, nazwa, bok_a, bok_b, bok_c):
        super().__init__(nazwa)
        self._bok_a = bok_a
        self._bok_b = bok_b
        self._bok_c = bok_c

    def pole(self):
        p = (self._bok_a + self._bok_b + self._bok_c) / 2
        return math.sqrt(p * (p - self._bok_a) * (p - self._bok_b) * (p - self._bok_c))

class TrojkatRownoboczny(Trójkąt):
    def __init__(self, nazwa, bok):
        super().__init__(nazwa, bok, bok, bok)


def main():
    figury = [
        Czworoscian("Czworoscian", 10),
        Szescian("Szescian", 5),
        Piramida("Piramida", 6),
    ]

    for figura in figury:
        print(f"{figura.nazwa}: {figura.pole()}")
main()
