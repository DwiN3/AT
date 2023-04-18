# Skończone

while True:
    n = int(input("Podaj liczbę poziomów -> "))
    if n % 2 == 0:
        znak = "*"
    else:
        znak = "#"
    for i in range(1, n + 1):
        print(" " * (n - i) + znak * (2 * i - 1))
    if n == 7:
        break