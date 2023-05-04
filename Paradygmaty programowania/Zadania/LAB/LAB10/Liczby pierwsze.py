# Skończone

def getPrimary(n):
    return list(filter(lambda x: all(x % i != 0 for i in range(2, x)), range(2, n)))

def main():
    print(getPrimary(99))

main()