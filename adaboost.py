import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Point:

    def __init__(self, x: float, y: float, label: float):
        self.x = x
        self.y = y
        self.label = int(label)

    def __repr__(self):
        return "X: {} , Y: {}, Label: {}".format(self.x, self.y, self.label)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x and self.y != other.y


class Rule:

    def __init__(self, coefficient, bias, weight=0):
        self.a = coefficient
        self.b = bias
        self.w = weight



def read(file: str):
    """
    This function receives a file and generates data points.
    :param file: given file to generate data points from.
    :return: list of points
    """
    points = []
    data = np.genfromtxt(file, delimiter=' ')
    np.random.shuffle(data)
    for line in data:
        points.append(Point(x=line[0], y=line[1], label=line[2]))
    return points


def generate(points: list):
    """
    This function receives a list of points and converts it to 4 different lists:
    - X and Y list of points for each label.
    :param points: list of points to convert
    :return: the generated lists
    """
    x1, y1, x2, y2 = ([] for _ in range(4))
    for pt in points:
        if pt.getLabel() == 1:
            x1.append(pt.x)
            y1.append(pt.y)
        else:
            x2.append(pt.x)
            y2.append(pt.y)
    return x1, y1, x2, y2


def createRules(points: list):
    """
    This function receives list of points and creates the rules.
    - Each rule is defined in the following way:
    - For every 2 points in the list, create a line y=ax+b.
    - The line is the rule.
    :param points: list of points
    :return: list of rules
    """
    rules = []
    for p1 in points:
        for p2 in points:
            if p1 != p2:
                incline = (p1.y - p2.y) / (p1.x - p2.x)
                bias = incline * p1.x * -1 + p1.y
                rules.append(Rule(coefficient=incline, bias=bias))
    return rules


def represent(points: list):
    """
    This function plots the data points.
    :param points: list of points to plot.
    :return: None
    """
    x1, y1, x2, y2 = generate(points)
    plt.scatter(x1, y1, color='red')
    plt.scatter(x2, y2, color='blue')

    # TODO - Add classifier lines
    plt.show()


represent(read('rectangle.txt'))