__author__ = 'vincent'

import numpy as np


class devide_struct:
    def __init__(self, feature, threshold, smaller, verbose):
        self.feature = feature
        self.threshold = threshold
        self.smaller = smaller
        self.verbose = verbose


def devide_data(filename, structs):
    x = []
    y = []
    z = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for index, l in enumerate(lines):
            chars = l.split(" ")
            xl = chars[1:]
            yl = chars[0]
            xl = [float(i) for i in xl]
            yl = [float(i) for i in yl]
            x.append(xl)
            y.append(yl)
            z.append(index)

    for i in structs:
        remain_x = []
        remain_y = []
        remain_z = []
        if i.smaller == True:
            for index in range(len(x)):
                if x[index][i.feature] < i.threshold:
                    remain_x.append(x[index])
                    remain_y.append(y[index])
                    remain_z.append(z[index])
        else:
            for index in range(len(x)):
                if x[index][i.feature] > i.threshold:
                    remain_x.append(x[index])
                    remain_y.append(y[index])
                    remain_z.append(z[index])
        x = remain_x
        y = remain_y
        z = remain_z
    return x, y, z


if __name__ == "__main__":
    structs = []
    structs.append(devide_struct(17, 0.5297, True, None))
    structs.append(devide_struct(9, 1.5472, True, None))
    structs.append(devide_struct(18, -5.7709, False, None))
    structs.append(devide_struct(7, -2.7890, False, None))
    structs.append(devide_struct(1, -4.0062, False, None))
    # structs.append(devide_struct(13, -5.4141, True, None))
    structs.append(devide_struct(10, 6.2304, False, None))

    import os
    os.chdir("/home/vincent/QtProjects/DecisionTree/test_data/Classification/")
    x, y, z = devide_data("test1.txt", structs)
    result0 = 0
    result1 = 0
    for i in range(len(y)):
        if y[i][0] == 1.0:
            result1 += 1
        else:
            result0 += 1
    print(" ".join([str(result0), str(result1)]))
    print(y)
    print(z)