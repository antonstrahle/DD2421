import numpy as np
import random , math
from scipy.optimize import minimize
from scipy.spatial import distance
import matplotlib.pyplot as plt

import data as dataset

from cvxopt.base import matrix
from cvxopt.solvers import qp


# ================================================== #
# Kernels
# ================================================== #

def radial_kernel(x, y):
    diff = np.subtract(x, y)
    return math.exp((-np.dot(diff, diff)) / (2 * SIGMA * SIGMA))

def polynomial_kernel(x, y):
	return np.power((np.dot(x, y) + 1), POLYNOMIAL_GRADE)

def linear_kernel(x, y):
    return np.dot(np.transpose(x), y)


# ================================================== #
# Optimization functions, not used siince problem with minimize
# ================================================== #

def zerofun(a):
	sum = 0.0
	for i in range(len(data)):
		sum += a[i]*(data[i])[2]
	return sum
    #return np.dot(a, np.transpose(data)[2])

def precompute(data, kernel):
    N = len(data)
    P = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            P[i][j] = (data[i])[2] * (data[j])[2] * kernel([(data[i])[0], (data[i])[1]], [(data[j])[0], (data[j])[1]])
    return P


#see eq.4
#this is what we want to optimize
def objective(a):
    sum_ij = 0.0
    sum_i = 0.0
    for i in range(0, len(P_matrix)):
        sum_i = sum_i + a[i]
        for j in range(0, len(P_matrix)):
            sum_ij = sum_ij + (a[i] * a[j] * P_matrix[i][j])
    return 0.5*sum_ij - sum_i

#not needed maybe?
#def compute_b(sv, target, nz_a, nz_Inputs, nz_Targets, kernel):
    #sum = 0.0
    #for i in range(0, len(nz_a)):
        #sum += nz_a[i] * nz_Targets[i] * kernel(sv, nz_Inputs[i])
    #return sum - target;

# ================================================== #
# Optimization using qp
# ================================================== #

#see https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf for omptimization using qp
def precompute(data, kernel, C):
    N = len(data)

    P = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            P[i][j] = (data[i])[2] * (data[j])[2] * kernel([(data[i])[0], (data[i])[1]], [(data[j])[0], (data[j])[1]])

    q = -np.ones((N, 1))
    G = -np.eye(N)
    h = np.zeros((N, 1))

    # Slack Variables
    if C != 0.:
        G = np.concatenate((np.eye(N), G))
        h = np.concatenate((C * np.ones((N, 1)), h))

    return P, q, h, G


def support_vectors(data, C):
	#N = len(data)
	#start = np.zeros(N)
	#B = [(0, C) for b in range(N)]
	#XC = {'type':'eq', 'fun': zerofun}
	#ret = minimize(objective, start, bounds=B, constraints=XC)
	
	#for some reason the minimize function does not work
	#I get a lower value of optimiste from the a's from the function qp
	#Will therefore use qp instead
	
	ret = qp(matrix(P_matrix), matrix(q), matrix(G), matrix(h))
	a = list(ret['x'])
	#print('[%s]' % ', '.join(map(str, a)))
	return nonzeroes(a, data)


def nonzeroes(a, data):
    nz_a = []
    svs = []
    for i in range(0, len(a)):
        if a[i] > 1.e-5 or a[i] < -1.e-5:
            nz_a.append(a[i])
            svs.append(((data[i])[0], (data[i])[1], (data[i])[2]))
    return nz_a, svs


# ================================================== #
# Plotting
# ================================================== #

def plot_data(classA, classB):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')
    
def plot_svs(svs):
    plt.plot([p[0] for p in svs], [p[1] for p in svs], marker="+", linestyle="None", markersize=15, color = 'black', mew=2)

def plot_boundaries(nz_a, svs, kernel):
    xgrid = np.linspace(-10, 10)
    ygrid = np.linspace(-4, 4)

    grid = np.array([[indicator(x, y, nz_a, svs, kernel) for x in xgrid] for y in ygrid])

    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1, 3, 1))



def indicator(x, y, nz_a, svs, kernel):
	##b = 0.0
	##for i in range(0, len(svs)):
		##b += nz_a[i] * (svs[i])[2] * kernel([(svs[0])[0], (svs[0])[1]], [(svs[i])[0], (svs[i])[1]]) - (svs[0])[2]
		
	ind = 0.0
	for i in range(0, len(svs)):
		ind += nz_a[i] * (svs[i])[2] *  kernel([x, y], [(svs[i])[0], (svs[i])[1]])

	#return ind - b 
	return ind 




# ================================================== #
# Main
# ================================================== #

#Inputs
POLYNOMIAL_GRADE = 2
SIGMA = 3
C = 10
#Different kernels
# - linear_kernel
# - polynomial_kernel
# - radial_kernel
kernel = polynomial_kernel

# ================================================== #
#generate data
random.seed(100)
classA, classB = dataset.generateData("1")
data = classA + classB
#random.shuffle(data)

# ================================================== #
#not used 
#P_matrix = precompute(data, kernel)

#computes whats needed for minimization which gives the a's which in turn gives the svs
P_matrix, q, h, G = precompute(data, kernel, C)
nz_a, svs = support_vectors(data, C)


# ================================================== #
# Show the plot
# ================================================== #

plot_data(classA, classB)
plot_svs(svs)
plot_boundaries(nz_a, svs, kernel)
plt.show()


#Radial for donuts with noise

kernel = radial_kernel

classA, classB = dataset.generateData("2")
data = classA + classB
#random.shuffle(data)

# ================================================== #
#not used 
#P_matrix = precompute(data, kernel)

#computes whats needed for minimization which gives the a's which in turn gives the svs
P_matrix, q, h, G = precompute(data, kernel, C)
nz_a, svs = support_vectors(data, C)


# ================================================== #
# Show the plot
# ================================================== #

plot_data(classA, classB)
plot_svs(svs)
plot_boundaries(nz_a, svs, kernel)
plt.show()


#Radial for half moons with noise

kernel = radial_kernel

classA, classB = dataset.generateData("3")
data = classA + classB
#random.shuffle(data)

# ================================================== #
#not used 
#P_matrix = precompute(data, kernel)

#computes whats needed for minimization which gives the a's which in turn gives the svs
P_matrix, q, h, G = precompute(data, kernel, C)
nz_a, svs = support_vectors(data, C)


# ================================================== #
# Show the plot
# ================================================== #

plot_data(classA, classB)
plot_svs(svs)
plot_boundaries(nz_a, svs, kernel)
plt.show()


# ================================================== #
# Test
# ================================================== #
#The first is from minimize from scipy, the second using qp. We get lower objective using the a's from qp
#The numbers are the a's
#print(objective([0.0, 0.0, 0.0, 0.733666512938, 0.0, 10.0, 0.0, 10.0, 1.73079424119e-13, 10.0, 1.29735766219e-13, 3.23526726723e-13, 0.0, 1.4867501951e-13, 0.346882234402, 1.59350592713, 0.0, 8.78088592511, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.67010842126e-12, 3.72877185484, 3.38406255676e-12, 0.0, 7.21988325021e-12, 0.0, 8.03215666849e-13, 2.34022336327e-12, 7.72616874474, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0]))
	  
#print(objective([3.9151805239128326e-08, 7.50534269335055e-08, 4.186617620877146e-08, 1.6741099140179116, 4.222015765807909e-08, 9.99999974163886, 4.355068226985404e-08, 9.999999946796779, 1.1752545364810855e-07, 9.999999975408896, 8.78195062563677e-08, 1.7913108985341396e-07, 3.540954083472783e-08, 2.414238787799132e-07, 1.4697683793979215, 2.431424601958541, 5.118784020399465e-08, 8.986289830305916, 4.624752950475128e-08, 5.921767201904342e-08, 1.3591069837816013e-07, 9.104416708898657e-08, 1.3681509987615035e-07, 3.4298321031401636e-08, 2.0846947787495605e-07, 2.1204088702516835, 4.3771173344148967e-07, 6.719570340886399e-08, 3.9595857741677554, 8.317242469419501e-08, 3.501199051072653e-07, 1.442879306926475e-07, 5.545666218183946, 9.999999947949538, 6.15696422405948e-08, 4.3352057261715735e-08, 9.999999844410619, 9.999999053129246, 4.3036473188270765e-08, 1.1344594486060692e-07]))




# ================================================================================= #
# Plots for report / JAN
# ================================================================================= #

random.seed(1)
classA, classB = dataset.generateData("1")
data = classA + classB
random.shuffle(data)
#Provided dataset not lineary separable




POLYNOMIAL_GRADE = 2
C = 0
kernel = polynomial_kernel


P_matrix, q, h, G = precompute(data, kernel, C)
nz_a, svs = support_vectors(data, C)

plot_svs(svs)
plot_data(classA, classB)
plot_boundaries(nz_a, svs, kernel)
plt.show()

#Inputs
POLYNOMIAL_GRADE = 3
C = 0
kernel = polynomial_kernel

P_matrix, q, h, G = precompute(data, kernel, C)
nz_a, svs = support_vectors(data, C)

plot_svs(svs)
plot_data(classA, classB)
plot_boundaries(nz_a, svs, kernel)
plt.show()


SIGMA = 3
C = 0
kernel = radial_kernel

P_matrix, q, h, G = precompute(data, kernel, C)
nz_a, svs = support_vectors(data, C)

plot_svs(svs)
plot_data(classA, classB)
plot_boundaries(nz_a, svs, kernel)
plt.show()

# ============================================================================================ #

#For showcasing the effect of sigma, radial kernel

SIGMA = 0.5
C = 0
kernel = radial_kernel

P_matrix, q, h, G = precompute(data, kernel, C)
nz_a, svs = support_vectors(data, C)

plot_svs(svs)
plot_data(classA, classB)
plot_boundaries(nz_a, svs, kernel)
plt.show()



SIGMA = 1
C = 0
kernel = radial_kernel

P_matrix, q, h, G = precompute(data, kernel, C)
nz_a, svs = support_vectors(data, C)

plot_svs(svs)
plot_data(classA, classB)
plot_boundaries(nz_a, svs, kernel)
plt.show()


SIGMA = 2
C = 0
kernel = radial_kernel

P_matrix, q, h, G = precompute(data, kernel, C)
nz_a, svs = support_vectors(data, C)

plot_svs(svs)
plot_data(classA, classB)
plot_boundaries(nz_a, svs, kernel)
plt.show()

# ============================================================================================ #


C = 1
kernel = linear_kernel


P_matrix, q, h, G = precompute(data, kernel, C)
nz_a, svs = support_vectors(data, C)

plot_svs(svs)
plot_data(classA, classB)
plot_boundaries(nz_a, svs, kernel)
plt.show()



