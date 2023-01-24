def findNumber(fileName):
    flag = 0
    num = []

    for c in fileName:
        if c == "_":
            flag = 1
        else:
            flag = 0

        if flag == 1:
            num.append(c)

