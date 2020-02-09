import numpy as np
import random , math
from scipy.optimize import minimize
from scipy.spatial import distance
import matplotlib.pyplot as plt

import data as dataset
#import inputagruments as ioargs


#input is (x_1, x_2) and target is -1 or 1
#inputs, targets, classA, classB = dataset.generateData(10, "1")

#plt.plot([p[0] for p in classA], [p[1] for p in classA], 'ro')
#plt.plot([p[0] for p in classB], [p[1] for p in classB], 'bo')
#plt.axis('equal')
#plt.show()

# ================================================== #
# Kernels
# ================================================== #

def radial_basis_function_kernel(x, y):
    return math.exp(-((distance.euclidean(x, y)**2)/(2*SIGMA*SIGMA)))

def polynomial_kernel(x, y):
	return np.power((np.dot(x, y) + 1), POLYNOMIAL_GRADE)
    #return (np.dot(np.transpose(x), y) + 1)**POLYNOMIAL_GRADE

def linear_kernel(x, y):
    return np.dot(np.transpose(x), y)

# ================================================== #
# Kernels
# ================================================== #


def zerofun(a):
    return np.dot(a, targets)

def precompute(P_matrix, inputs, kernel, targets):
    for i in range(0, N):
        for j in range(0, N):
            P_matrix[i][j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])
    return P_matrix


#see eq.4
def objective(a):
    sum_ij = 0.0
    sum_i = 0.0
    for i in range(0, len(P_matrix)):
        sum_i = sum_i + a[i]
        for j in range(0, len(P_matrix[0])):
            sum_ij = sum_ij + (a[i] * a[j] * P_matrix[i][j])
    return 0.5*sum_ij - sum_i

def compute_b(sv, target, nz_a, nz_Inputs, nz_Targets, kernel):
    sum = 0.0
    for i in range(0, len(nz_a)):
        sum += nz_a[i] * nz_Targets[i] * kernel(sv, nz_Inputs[i])
    return sum - target;

def ind(x, y, nz_a, nz_Inputs, nz_Targets, kernel):
    sum = 0.0
    #b = compute_b(nz_Inputs[1], nz_Targets[1], nz_a, nz_Inputs, nz_Targets, kernel)
    for i in range(0, len(nz_a)):
        sum += nz_a[i] * nz_Targets[i] * kernel([x, y], nz_Inputs[i])

    #return sum - b
    return sum 

# plots our data
# the boundary only depend on the non-zero values, aka the support vectors
def plot(nz_a, nz_Inputs, nz_Targets, kernel):
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    # check which kernel we are using
    if kernel == "linear":
        grid = np.array([[ind(x, y, nz_a, nz_Inputs, nz_Targets, linear_kernel)
                        for x in xgrid]
                        for y in ygrid])
    elif kernel == "polynomial":
        grid = np.array([[ind(x, y, nz_a, nz_Inputs, nz_Targets, polynomial_kernel)
                        for x in xgrid]
                        for y in ygrid])
    else:
        grid = np.array([[ind(x, y, nz_a, nz_Inputs, nz_Targets, radial_basis_function_kernel)
                        for x in xgrid]
                        for y in ygrid])

    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1, 3, 1))

    plt.show()
    

def nonzeroes(a, inputs, targets):
  nz_a = []
  nz_Inputs = []
  nz_Targets = []
  for i in range(0, len(a)):
      if a[i] > 1.e-5 or a[i] < -1.e-5:
        nz_a.append(a[i])
        nz_Inputs.append(inputs[i])
        nz_Targets.append(targets[i])
  return nz_a, nz_Inputs, nz_Targets


#Inputs
N = 80
C = 1
POLYNOMIAL_GRADE = 2
SIGMA = 1
np.random.seed(100)
inputs, targets, classA, classB = dataset.generateData(N, "1")
kernel = "linear"

start = np.zeros(N)
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun': zerofun}
P_matrix = [[0 for i in range(0, N)] for j in range(0, N)]

if kernel == "linear":
	P_matrix = precompute(P_matrix, inputs, linear_kernel, targets)
elif kernel == "polynomial":
	P_matrix = precompute(P_matrix, inputs, polynomial_kernel, targets)
elif kernel == "radial":
	P_matrix = precompute(P_matrix, inputs, radial_basis_function_kernel, targets)

ret = minimize(objective, start, bounds=B, constraints=XC)
a = ret['x']
success = ret['success']

if success == True:
	print ("Optimal solution found")
else:
	print ("No optimal solution found")


nz_a, nz_Inputs, nz_Targets = nonzeroes(a, inputs, targets)



#support vectors marked with "+"
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'ro')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'bo')
plt.plot([p[0] for p in nz_Inputs], [p[1] for p in nz_Inputs], marker="+", linestyle="None", markersize=15, color = 'black', mew=2)
plt.axis('equal')

plot(nz_a, nz_Inputs, nz_Targets, "linear")





