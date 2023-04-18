# Skończone

zdanie = input("Podaj zdanie -> ")
litera_a = 0

for litera in zdanie:
    if litera == 'a' or litera == 'A':
        litera_a += 1
print("Liczba wystąpień litery 'a' w zdaniu = ", litera_a)