# Skończone

import random

wylosowana_liczba = random.randint(1, 10)

liczba_prob = 3

while liczba_prob > 0:
  wpisana_liczba = int(input("Podaj liczbę od 1 do 10 -> "))

  if wpisana_liczba == wylosowana_liczba:
    print("Brawo, zgadłeś!")
    break

  elif wpisana_liczba < wylosowana_liczba:
    print("Podana liczba jest za mała.\n")
    liczba_prob -= 1

  else:
    print("Podana liczba jest za duża.\n")
    liczba_prob -= 1

else:
  print("Niestety, nie udało Ci się zgadnąć. Szukana liczba to",
        wylosowana_liczba)