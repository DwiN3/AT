# Skończone

for liczba in range(2, 101):
    for dzielnik in range(2, liczba):
        if liczba % dzielnik == 0:
            break
    else:
        print(liczba)