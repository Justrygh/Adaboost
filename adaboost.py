import matplotlib.pyplot as plt
import numpy as np
import math
import statistics


class Point:

    def __init__(self, x: float, y: float, label: int):
        """
        :param x: X-axis
        :param y: Y-axis
        :param label: The given data point label
        """
        self.x = x
        self.y = y
        self.label = label
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

    def __init__(self, p1: Point,p2: Point,):
        """
        - Rule: y=ax+b
        :param point: Single point for computing line equation.
        :param coefficient: a
        :param bias: b
        """
        self.p1 = p1
        self.p2 = p2
        self.a = (self.p1.y - self.p2.y)
        self.b = (self.p2.x - self.p1.x)
        self.c = (self.p1.x * self.p2.y - self.p2.x * self.p1.y)
        self.label = p1.label if p1.label == p2.label else 1

    def eval(self, point: Point):
        """
        This function evaluates the following equation:
        - eval = ax+b-y
        - if eval > 0 return 1 otherwise -1
        :return: eval
        """
        if self.a * point.x + self.b *point.y + self.c >= 0:
            return self.label
        else:
            return -self.label


def read_data(file: str):
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


def create_rules(points: list):
    """
    This function receives a list of data points and creates the rules.
    - Each rule is defined in the following way:
    - For every 2 points in the list, create a line y=ax+b.
    - The line is the rule.
    :param points: list of data points
    :return: list of rules
    """
    rules = []
    length = len(points)
    for i in range(length):
        p1 = points[i]
        for j in range(i + 1, length):
            p2 = points[j]
            rules.append(Rule(p1=p1,p2=p2))
            # else:
            # if p1.x equals p2.x -> incline = coefficient = infinity & bias = 0
            # rules.append(Rule(point=p1, coefficient=np.inf))
    return rules


def split_data(points: list):
    """
    This function receives a list of data points and splits it to 2 equal groups (train & test).
    :param points: list of data points
    :return: train & test groups.
    """
    if len(points) % 2 == 0:
        return points[:int(len(points) / 2)], points[int(len(points) / 2):]
    else:
        return points[:int((len(points) + 1) / 2)], points[int((len(points) + 1) / 2):]


def predict_value(rules_with_weights: list, point: Point):
    """
    This function predict the label of a point based on rules
    :param rules_with_weights: list of important rules.
    :param point: to predict
    :return: 1 if predict 1 else -1
    """
    sum = 0

    for h in rules_with_weights:
        sum += h[1] * h[0].eval(point)

    return 1 if sum > 0 else -1


def calculate_point_error(rules_with_weights: list, point: Point):
    """
    This function calculates the empirical error on given point
    :param rules_with_weights: list of important rules.
    :param point: to check for error
    :return: 1 if there is error else 0
    """
    return 1 if predict_value(rules_with_weights, point) != point.label else 0


def calculate_list_error(rules_with_weights: list, l: list):
    """
      This function calculates the empirical error on given list
      :param rules_with_weights: list of important rules.
      :param l: list of data points (training set)
      :return: total error
      """
    error_sum = 0

    for p in l:
        error_sum += calculate_point_error(rules_with_weights, p)

    return error_sum / len(l)


def calculate_error(rules_with_weights: list, train: list, test: list, iterations: int):
    """
    This function calculates the empirical error on the training and test sets.
    :param rules_with_weights: list of important rules.
    :param train: list of data points (training set)
    :param test: list of data points (testing set)
    :param iterations: number of iterations for computing the empirical errors
    :return: lists of empirical errors on the training and testing set over k iterations.

    - NOTE: This function was CHANGED!

    """
    train_errors, test_errors = ([] for _ in range(2))
    iterations = len(rules_with_weights) if iterations > len(rules_with_weights) else iterations

    for i in range(iterations):
        train_errors.append(calculate_list_error(rules_with_weights[:i + 1], train))
        test_errors.append(calculate_list_error(rules_with_weights[:i + 1], test))

    return [train_errors, test_errors]


def run(points: list, rules: list, iterations: int):
    """
    This function simulates a single run of Adaboost algorithm.
    :param points: list of data points
    :param iterations: number of iterations to perform the algorithm.
    :return:
    """
    np.random.shuffle(points)

    train, test = split_data(points)

    for pt in train:
        pt.w = 1 / len(train)  # Initialize point weights

    rules_with_weights = []
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

        classifier_weight = math.log((1 - min_error) / min_error, math.e) / 2  # Update classifier weight based on error
        normalization = 0

        for pt in train:
            """ Calculate the normalizing constant - save the calculation of new point weights in placeholder """
            power = classifier_weight * classifier.eval(point=pt) * pt.label * -1
            pt.placeholder = pt.w * (math.e ** power)
            normalization += pt.placeholder

        for pt in train:
            weight = (1 / normalization) * pt.placeholder
            pt.w = weight

        rules_with_weights.append((classifier, classifier_weight))
        represent_data_points(train, rules_with_weights)

    return calculate_error(rules_with_weights=rules_with_weights, train=train, test=test, iterations=iterations)

    # TODO - Return the empirical error of the function H on the training set and on the test set.


def generate_data_lists(points: list):
    """
    This function receives a list of data points and converts it to 4 different lists:
    - X and Y list of data points for each label.
    :param points: list of data points to convert
    :return: the generated lists
    """
    x1, y1, x2, y2 = ([] for _ in range(4))
    for pt in points:
        if pt.label == 1:
            x1.append(pt.x)
            y1.append(pt.y)
        else:
            x2.append(pt.x)
            y2.append(pt.y)
    return x1, y1, x2, y2


def represent_data_points(points: list,rules_with_weights: list):
    """
    This function plots the data points.
    :param points: list of data points to plot.
    :return: None
    """
    x1, y1, x2, y2 = generate_data_lists(points)
    plt.scatter(x1, y1, color='red')
    plt.scatter(x2, y2, color='blue')


    for rule in rules_with_weights:
        r = rule[0]
        plt.plot([r.p1.x, r.p1.y],[r.p2.x, r.p2.y], marker='o')

    # TODO - Add classifier lines

    plt.show()


points = read_data('rectangle.txt')
rules = create_rules(points)
iterations = 8
rounds = 10

train_errors = [[0 for i in range(rounds)] for j in range(iterations)]
test_errors = [[0 for i in range(rounds)] for j in range(iterations)]

for i in range(rounds):
    [train_error, test_error] = run(points, rules, iterations)

    for j in range(iterations):
        train_errors[j][i] = train_error[j]
        test_errors[j][i] = test_error[j]

    print(i)

for i in range(iterations):
    print("k = ", (i + 1), " train error: ", "%.3f" % statistics.mean(train_errors[i]), "test error: ",
          "%.3f" % statistics.mean(test_errors[i]))
