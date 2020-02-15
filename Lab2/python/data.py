import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def basic_dataset():
    # Not Linearly separable points
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(10)] +\
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(10)]

    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(20)]

    return classA, classB

def make_circles(rmin, rmax, n, thetaMax, classification):
    
    #added gaussian noise to radius
    
    polar = [(random.uniform(np.max(rmin + random.normalvariate(0, 0.25), 0), 
                             np.max(rmax + random.normalvariate(0, 0.25), 0)), 
              random.uniform(0,thetaMax)) for i in range (n)]
    
    cartesian = []
    
    for i in range (n):
        
        x = polar[i][0]*np.cos(polar[i][1])
        
        y = polar[i][0]*np.sin(polar[i][1])
        
        cartesian.append((x,y, classification))

    return cartesian


def radial_dataset(thetaMax):
    
    classA = make_circles(0, 2, 20, thetaMax, 1)
    classB = make_circles(2, 5, 20, thetaMax, -1)
    
    return classA, classB


def generateData(dataset):
    
    if dataset == "1":
        classA, classB = basic_dataset()
        
    elif dataset == "2":
        classA, classB = radial_dataset(360)
    
    elif dataset == "3":
        classA, classB = radial_dataset(180)
    
    data = classA + classB
    return classA, classB













