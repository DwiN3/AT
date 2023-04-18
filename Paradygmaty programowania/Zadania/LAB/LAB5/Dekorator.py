# Skończone

from time import perf_counter

def calculate_time(func):
  def wrapper(*args, **kwargs):
    start = perf_counter()
    result = func(*args, **kwargs)
    end = perf_counter()
    time = end - start
    print(f"Czas wykonania funkcji {func.__name__}: {time:.6f} s")
    return result
  return wrapper

@calculate_time
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
    my_range(1.1, 1, 0.5)
    my_range(1.1, 10, 0.5)
    my_range(1.1, 100,0.5)
    my_range(1.1, 1000,0.5)
    my_range(1.1, 10000,0.5)
    my_range(1.1, 100000,0.5)
  except ValueError as error:
    print(error)
main()