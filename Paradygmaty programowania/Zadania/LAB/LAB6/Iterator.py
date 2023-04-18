# Skończone

class MyRange:
    def __init__(self, *args):
        len_args = len(args)
        if len_args == 0 or len_args > 3:
            raise ValueError("MyRange przyjmuje od 1 do 3 argumentów")
        elif len_args == 1:
            self.a = 0.0
            self.b = args[0]
            self.k = 1.0
        elif len_args == 2:
            self.a = args[0]
            self.b = args[1]
            self.k = 1.0
        elif len_args == 3:
            self.a = args[0]
            self.b = args[1]
            self.k = args[2]
        if self.k == 0.0:
            raise ValueError("MyRange k jest równe 0")
        self.current = self.a

    def __iter__(self):
        return self

    def __next__(self):
        if (self.k > 0 and self.current >= self.b) or (self.k < 0 and self.current <= self.b):
            raise StopIteration
        result = self.current
        self.current += self.k
        return result

def main():
  try:
    for number in MyRange(1.1, 2.2, 0.5):
      print(number)
  except ValueError as error:
    print(error)
main()