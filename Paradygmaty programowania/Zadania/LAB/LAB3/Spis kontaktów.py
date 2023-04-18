# Skończone

kontakty = {
  ('Jan', 'Kowalski'): "123456789",
  ('Adam', 'Nowak'): "987654321",
  ('Adam', 'Kowalski'): "600300900"
}


def wyswietl_numer(nazwisko):
  numery = []
  for kontakt, numer in kontakty.items():
    if kontakt[1] == nazwisko:
      numery.append(numer)
  return numery


print(kontakty['Jan', 'Kowalski'])
print(wyswietl_numer('Kowalski'))