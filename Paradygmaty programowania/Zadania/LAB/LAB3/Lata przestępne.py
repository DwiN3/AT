# Skończone

def lata_przestepne(rok_poczatkowy, rok_koncowy):
    return [rok for rok in range(rok_poczatkowy, rok_koncowy + 1) if (rok % 4 == 0 and rok % 100 != 0) or rok % 400 == 0]

print(lata_przestepne(1900, 2000))