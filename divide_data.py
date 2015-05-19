__author__ = 'vincenti

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
    with open(filename, "r") as f:
        lines = f.readlines()
        for index, l in enumerate(lines):
            chars = l.split("0")
            xl = chars[1:]
            yl = chars[0]
            x.append(xl)
            y.append(yl)

    remain = [x, y]
    for i in structs:
        xl = []
        if i.smaller == True:
            for x in i[0]:
                for row in x:
                    if x[row][i.feature] < i.threshold:
                        xl.
