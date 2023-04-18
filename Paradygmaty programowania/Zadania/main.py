# Skończone

def liczba_z_napisu(napis):
  liczby = {
    'jeden': 1,
    'dwa': 2,
    'trzy': 3,
    'cztery': 4,
    'pięć': 5,
    'sześć': 6,
    'siedem': 7,
    'osiem': 8,
    'dziewięć': 9
  }
  dziesiatki = {
    'dwadzieścia': 20,
    'trzydzieści': 30,
    'czterdzieści': 40,
    'pięćdziesiąt': 50
  }

  slowa = napis.split()
  dziesiatka, jednostka = slowa
  return dziesiatki[dziesiatka] + liczby[jednostka]


def zamien_na_liczbe(napis):
  slowa = napis.split()
  if len(slowa) != 2:
    raise ValueError("Niepoprawny napis")

  liczba = liczba_z_napisu(napis)
  return liczba


print(zamien_na_liczbe('dwadzieścia jeden'))