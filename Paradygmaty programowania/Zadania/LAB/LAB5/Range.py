# Skończone

def my_range(*args):
  len_args = len(args)
  if len_args == 0 or len_args > 3:
    raise ValueError("my_range() przyjmuje od 1 do 3 argumentów")
  elif len_args == 1:
    a = 0.0
    b = args[0]
    k = 1.0
  elif len_args == 2:
    a = args[0]
    b = args[1]
    k = 1.0
  elif len_args == 3:
    a = args[0]
    b = args[1]
    k = args[2]
  if k == 0.0:
    raise ValueError("my_range() k jest równe 0")
  results = []
  while (k > 0 and a < b) or (k < 0 and a > b):
    results.append(a)
    a+=k
  return results

def main():
  try:
    print(my_range(1.1, 2.2, 0.5))
    print(my_range(1.1, 2.1, 0.5))
    print(my_range(1.1, 2.2))
    print(my_range(2.2))
    print(my_range(7.8, 5.6, -1.2))
  except ValueError as error:
    print(error)
  
  # test dla kroku o wartości 0
  try:
    print(my_range(7.8, 5.6, 0))
  except ValueError as error:
    print(error)  
  
  # test dla 4 parametrów
  try:
    print(my_range(1.1, 2.2, 0.5, 4.4))
  except ValueError as error:
    print(error)
main()