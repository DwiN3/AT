# Skończone

licznik_kombinacji = 0
for m2 in range(0, 101, 2):
    for m5 in range(0, 101-m2, 5):
        for m10 in range(0, 101-m2-m5, 10):
            if m2 + m5 + m10 == 100:
                licznik_kombinacji += 1
print("Liczba możliwych kombinacji = ", licznik_kombinacji)