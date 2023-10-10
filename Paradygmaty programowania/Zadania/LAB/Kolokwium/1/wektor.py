class Vector:
    def __init__(self, *args):
        self.wektor = []
        if len(args) == 1:
            self.wektor = args[0]

        elif len(args) == 2:
            ile = args[0]
            co = args[1]
            while ile:
                self.wektor.append(co)
                ile -=1
        else:
            raise ValueError("Błędna ilość argumentów")

    def __str__(self):
        return f"{self.wektor}"

    def __add__(self, other):
        sum_list =[]
        dlugosc = len(self.wektor)
        for element in range(dlugosc):
            sum_list.append(self.wektor[element] + other.wektor[element])
        return Vector(sum_list)

    def __mul__(self, other):
        sum = 0
        dlugosc = len(self.wektor)
        for element in range(dlugosc):
            sum += self.wektor[element] * other.wektor[element]
        return sum

def main():
    try:
        v1 = Vector([1, 2, 3])
        v2 = Vector(3, 1)
        print(v1)
        print(v2)
        print(v1 + v2)
        print(v1 * v2)
    except ValueError as error:
        print(error)
main()