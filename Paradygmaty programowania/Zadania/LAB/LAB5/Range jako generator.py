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
  while (k > 0 and a < b) or (k < 0 and a > b):
      yield a
      a+=k

def main():
  try:
    for number in my_range(1.1, 2.2, 0.5):
      print(number)
  except ValueError as error:
    print(error)
main()