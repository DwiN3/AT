# Skończone

def referenceSumNormal(list):
        if not list:
            return 0;
        return list[0] + referenceSumNormal(list[1:])

def referenceSumTail(list, acc=0):
    if not list:
        return acc
    return referenceSumTail(list[1:], acc + list[0])

def main():
    list1 = [1,2,3,4]
    list2 = []
    print(referenceSumNormal(list1))
    print(referenceSumNormal(list2))
    print(referenceSumTail(list1))
    print(referenceSumTail(list2))
main()