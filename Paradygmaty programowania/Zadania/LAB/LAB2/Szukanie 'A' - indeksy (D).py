# Skończone

zdanie = input("Podaj zdanie -> ")

for index, litera in enumerate(zdanie):
  if litera == 'a' or litera == 'A':
    print(f"Litera 'a' znaleziona na indeksie {index}")