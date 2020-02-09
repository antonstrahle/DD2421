import numpy , random
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def basic_dataset():
    # Not Linearly separable points
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(10)] +\
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(10)]

    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(20)]

    return classA, classB


def generateData(dataset):
    if dataset == "1":
        classA, classB = basic_dataset()

    data = classA + classB
    return classA, classB













