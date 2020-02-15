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

# ================================================== #
# Helper for radial datasets
# ================================================== #

def make_circles(rmin, rmax, n, thetaMax, classification, osx = 0):
    
    #added gaussian noise to radius
    
    polar = [(random.uniform(np.max(rmin + random.normalvariate(0, 0.25), 0), 
                             np.max(rmax + random.normalvariate(0, 0.25), 0)), 
              random.uniform(0,thetaMax)) for i in range (n)]
    
    cartesian = []
    
    for i in range (n):
        
        x = polar[i][0]*np.cos(np.deg2rad(polar[i][1])) + random.normalvariate(osx, 0.25)
        
        y = polar[i][0]*np.sin(np.deg2rad(polar[i][1])) + random.normalvariate(0, 0.25)
        
        cartesian.append((x,y, classification))

    return cartesian

# ================================================== #
# Creates a radial dataset with a specific max angle 
# to create donuts, moons, etc
# ================================================== #

def radial_dataset(thetaMax, osAx = 0):
    
    classA = make_circles(0, 3, 20, thetaMax, 1, osAx) 
    classB = make_circles(3, 5, 20, thetaMax, -1)
    
    return classA, classB


def generateData(dataset):
    
    if dataset == "1":
        classA, classB = basic_dataset()
        
    elif dataset == "2":
        classA, classB = radial_dataset(360)
    
    elif dataset == "3":
        classA, classB = radial_dataset(180, 2)
    
    data = classA + classB
    return classA, classB










