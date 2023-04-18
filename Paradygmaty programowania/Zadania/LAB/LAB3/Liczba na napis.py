# Skończone

def zamien_na_slowa(napis):
    slownik = {'0': "zero", '1': "jeden", '2': "dwa", '3': "trzy", '4': "cztery", '5': "pięć", '6': "sześć",
               '7': "siedem", '8': "osiem", '9': "dziewięć"}
    slowa = []
    for znak in napis:
        if znak.isdigit():
            slowa.append(slownik[znak])
    wszystkie_slowa = ' '.join(slowa)
    return wszystkie_slowa


print(zamien_na_slowa('1a410'))