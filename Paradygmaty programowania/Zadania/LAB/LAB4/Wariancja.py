# Skończone

def mean_and_variance(tab):
  N = len(tab)
  x_ = sum(tab) / N
  w = sum((xi - x_)**2 for xi in tab) / (N - 1)
  return x_, w


tab1 = []
while True:
  liczba = int(input("Podaj liczbę -> "))
  if liczba == 0:
    print("\n")
    break
  tab1.append(liczba)

średnia1, wariancja1 = mean_and_variance(tab1)
print(tab1)
print(f"Średnia wynosi:   {średnia1:.3f}")
print(f"Wariancja wynosi: {wariancja1:.3f}\n")

tab2 = [3, 3, 3, 3]
średnia2, wariancja2 = mean_and_variance(tab2)
print(tab2)
print(f"Średnia wynosi:   {średnia2:.3f}")
print(f"Wariancja wynosi: {wariancja2:.3f}\n")

tab3 = [5, 6, 7, 8, 9]
średnia3, wariancja3 = mean_and_variance(tab3)
print(tab3)
print(f"Średnia wynosi:   {średnia3:.3f}")
print(f"Wariancja wynosi: {wariancja3:.3f}\n")