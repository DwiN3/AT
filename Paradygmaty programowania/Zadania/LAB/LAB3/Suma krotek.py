# Skończone

def sumuj_krotki(krotki):
  suma = []
  for krotka in krotki:
    if isinstance(krotka[0], str) and isinstance(krotka[1], str):
      suma.append(krotka[0] + krotka[1])
    else:
      suma.append(krotka[0] + krotka[1])
  return suma


lista1 = [(1, 2), (3, 4), (5, 6), (7, 8)]
print(sumuj_krotki(lista1))

lista2 = [('a', 'b'), ('a', 'c'), ('b', 'c')]
print(sumuj_krotki(lista2))
