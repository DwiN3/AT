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
        return sum(sciana.pole() for sciana in self._sciany)

class Czworoscian(Mixin, Kształt):
    def __init__(self, nazwa, bok):
        super().__init__(nazwa)
        self._sciany = [TrojkatRownoboczny(f"sciana{i}", bok) for i in range(1, 5)]

class Szescian(Mixin, Kształt):
    def __init__(self, nazwa, bok):
        super().__init__(nazwa)
        self._sciany = [Kwadrat(f"sciana{i}", bok) for i in range(1, 7)]

class Piramida(Mixin, Kształt):
    def __init__(self, nazwa, bok):
        super().__init__(nazwa)
        self._sciany = [TrojkatRownoboczny(f"sciana{i}", bok) for i in range(1, 5)]

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
        Czworoscian("Czworościan", 12),
        Szescian("Sześcian", 8),
        Piramida("Piramida", 4),
    ]

    for figura in figury:
        print(f"{figura.nazwa}: {figura.pole()}")
main()