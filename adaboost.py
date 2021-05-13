import matplotlib.pyplot as plt
import numpy as np
import math


class Point:

    def __init__(self, x: float, y: float, label: int):
        """
        :param x: X-axis
        :param y: Y-axis
        :param label: The given data point label
        """
        self.x = x
        self.y = y
        self.label = label.astype(np.int)
        self.w = 0
        self.placeholder = None

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x and self.y != other.y

    def toArray(self):
        """
        This function converts a single data point into array: [x, y, label]
        :return: array
        """
        return [self.x, self.y, self.label]


class Rule:

    def __init__(self, point: Point, coefficient, bias=0):
        """
        - Rule: y=ax+b
        :param point: Single point for computing line equation.
        :param coefficient: a
        :param bias: b
        """
        self.p = point
        self.a = coefficient
        self.b = bias
        self.w = 0

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

    def __ne__(self, other):
        return self.a != other.a and self.b != other.b

    def eval(self, point: Point):
        """
        This function evaluates the following equation:
        - eval = ax+b-y
        - if eval > 0 return 1 otherwise -1
        :return: eval
        """
        if self.a * self.p.x + self.b - self.p.y > 0:
            return 1
        else:
            return -1


def read(file: str):
    """
    This function receives a file, reads the data from the given file, shuffles the data and generates data points.
    :param file: given file to generate data points from.
    :return: list of data points
    """
    points = []
    data = np.genfromtxt(file, delimiter=' ')
    np.random.shuffle(data)
    for line in data:
        points.append(Point(x=line[0], y=line[1], label=line[2]))
    return points


def createRules(points: list):
    """
    This function receives a list of data points and creates the rules.
    - Each rule is defined in the following way:
    - For every 2 points in the list, create a line y=ax+b.
    - The line is the rule.
    :param points: list of data points
    :return: list of rules
    """
    rules = []
    for p1 in points:
        for p2 in points:
            if p1 != p2:
                # if p1.x != p2.x:
                incline = (p1.y - p2.y) / (p1.x - p2.x)
                bias = p1.y - incline * p1.x
                rules.append(Rule(point=p1, coefficient=incline, bias=bias))
                # else:
                # if p1.x equals p2.x -> incline = coefficient = infinity & bias = 0
                # rules.append(Rule(point=p1, coefficient=np.inf))
    return rules


def split(points: list):
    """
    This function receives a list of data points, shuffles the data and splits it to 2 equal groups (train & test).
    :param points: list of data points
    :return: train & test groups.
    """
    if len(points) % 2 == 0:
        return points[:int(len(points)/2)], points[int(len(points)/2):]
    else:
        return points[:int((len(points)+1)/2)], points[int((len(points)+1)/2):]


def calculateError(rules: list, train: list, test: list):
    """
    This function calculates the empirical error on the training and test sets.
    :param rules: list of important rules.
    :param train: list of data points (training set)
    :param test: list of data points (testing set)
    :return: lists of empirical errors on the training and testing set over k iterations.
    """
    train_error, test_error = ([] for _ in range(2))
    for i in range(len(rules)):
        pass


def run(points: list, rules: list, iterations: int):
    """
    This function simulates a single run of Adaboost algorithm.
    :param points: list of data points
    :param iterations: number of iterations to perform the algorithm.
    :return:
    """
    for pt in points:
        pt.w = 1/len(points)  # Initialize point weights
    train, test = split(points)

    important_rules = []
    for i in range(iterations):
        min_error = np.inf  # Find the min error each iteration and the classifier.
        classifier = None
        for h in rules:
            error = 0
            for pt in train:
                predict = h.eval(point=pt)
                # TODO - Check prediction conditions: how to classify between 1 and -1
                predict = -1 if predict < 0 else 1
                if predict != pt.label:
                    error += pt.w

            if error < min_error:  # Find min. error classifier
                min_error = error
                classifier = h

        classifier.w = math.log((1-min_error)/min_error, math.e) / 2  # Update classifier weight based on error
        normalization = 0

        for pt in train:
            """ Calculate the normalizing constant - save the calculation of new point weights in placeholder """
            power = classifier.w * classifier.eval(point=pt) * pt.label * -1
            pt.placeholder = pt.w * (math.e ** power)
            normalization += pt.placeholder

        for pt in train:
            weight = (1 / normalization) * pt.placeholder
            pt.w = weight

        important_rules.append(classifier)

    calculateError(rules=important_rules, train=train, test=test)

    # TODO - Return the empirical error of the function H on the training set and on the test set.


def generate(points: list):
    """
    This function receives a list of data points and converts it to 4 different lists:
    - X and Y list of data points for each label.
    :param points: list of data points to convert
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


def represent(points: list):
    """
    This function plots the data points.
    :param points: list of data points to plot.
    :return: None
    """
    x1, y1, x2, y2 = generate(points)
    plt.scatter(x1, y1, color='red')
    plt.scatter(x2, y2, color='blue')

    # TODO - Add classifier lines
    plt.show()


points = read('rectangle.txt')
rules = createRules(points)
iterations = 8
run(points, rules, iterations)