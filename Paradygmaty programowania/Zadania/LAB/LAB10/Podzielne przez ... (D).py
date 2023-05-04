# Skończone

def podzielnePrzez2do11(od, do):
    return list(filter(lambda x: any(x % n == 0 for n in range(2, 12)), range(od, do+1)))

def main():
    print(podzielnePrzez2do11(1, 50))

main()