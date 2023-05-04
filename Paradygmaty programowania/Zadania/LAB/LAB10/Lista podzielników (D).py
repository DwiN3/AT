# Skończone

def maksymalnyPodzielniki(od, do):
    dzielniki = list(filter(lambda x: x > 0, range(1, 10)))
    return list(map(lambda x: max(filter(lambda y: x % y == 0, dzielniki)), range(od, do+1)))

def main():
    print(maksymalnyPodzielniki(1, 20))

main()