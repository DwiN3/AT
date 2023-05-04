# Skończone

def getPrimary(n):
    return list(filter(lambda x: all(map(lambda i: x % i != 0, range(2, x))), range(2, n)))

def main():
    print(getPrimary(99))

main()