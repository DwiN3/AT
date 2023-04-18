# Skończone

def common_elements(obj1, obj2):
  lista = list(set([elem for elem in obj1 if elem in obj2]))
  return lista

def main():
  list1 = [1, 2, 3, 4]
  list2 = [3, 4, 3, 9]
  print(common_elements(list1, list2))

  krotka1 = (3, 3, 4, 5)
  krotka2 = (3, 3, 2, 5)
  print(common_elements(krotka1, krotka2))

  slowo1 = 'To przykladowe słowo'
  slowo2 = 'przykład nr 2'
  print(common_elements(slowo1, slowo2))
main()

