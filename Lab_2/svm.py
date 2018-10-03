import random, math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import kernels as krnl
from generate_data import *

 # ============================================ #
 #  Def global values
 # ============================================ #

# Set kernel
kernel = krnl.linear_kernel
inData = getSlackData()
slack = 1

classA, classB, data, t = inData

N = data.shape[0]

P = np.zeros( (N, N) )

 # ============================================ #
 #  def functions
 # ============================================ #

def precompute() :
    for i in range(N) :
        for j in range(N) :
            P[i][j] = t[i] * t[j] * kernel([data[i][0], data[i][1]], [data[j][0], data[j][1]])

def objective(alpha) :
    y = 0.0
    # y = np.dot(alpha, np.dot(alpha, P))
    for i in range(N) :
        for j in range(N) :
            y += alpha[i] * alpha[j] * P[i][j]
    y = 0.5 * y
    y -= np.sum(alpha)

    return y

def zerofun(a) : 
    return np.dot(a, t)

def getb(inData) : # inData: [alpha, supportX, supportY, supportT]
    b = 0.0
    for i in range(len(inData[0])) :
        b += inData[0][i] * inData[3][i] * kernel([inData[1][0], inData[2][0]], [inData[1][i], inData[2][i]])

    b -= inData[3][0]
    return b

def ind(x, y, inData) : # inData: [alpha, supportX, supportY, supportT]
    ind = 0.0
    for i in range(len(inData[0])) :
        ind += inData[0][i] * inData[3][i] * kernel([inData[1][i], inData[2][i]], [x, y])

    ind -= b

    return ind

 # ============================================ #
 #  Run code
 # ============================================ #

XC = {'type':'eq', 'fun':zerofun}
start = np.zeros(N)

B = [(0, slack) for b in range(N)]

precompute()

ret = minimize(objective, start, bounds=B, constraints=XC)
alphas = ret['x']
success = ret['success']

if success == True :
    print("Yay sucess!!!")
else :
    print("Awww.... fail...")


supportVectorsX = list()
supportVectorsY = list()
supportVectorsT = list()
savedAlphas = list()

for i in range(alphas.size) :
    if alphas[i] > 1e-5 :
        savedAlphas.append(alphas[i])
        supportVectorsX.append(data[i][0])
        supportVectorsY.append(data[i][1])
        supportVectorsT.append(t[i])


indData = [savedAlphas, supportVectorsX, supportVectorsY, supportVectorsT]
b = getb(indData)

print(savedAlphas)
print(supportVectorsX)
print(supportVectorsY)
print(supportVectorsT)

 # ============================================ #
 #  Plot data points & decision boundary
 # ============================================ #

plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)

grid = np.array([[ind(x, y, indData) for x in xgrid] for y in ygrid]
)

plt.contour(xgrid, ygrid, grid, 
    (-1.0, 0.0, 1.0), 
    colors = ('red', 'black', 'blue'), 
    linewidths = (1, 3, 1)
    )

plt.axis('equal') # Force same scale on both axes
plt.savefig('svmplot.pdf') # Save a copy in a file
plt.show() # Show the plot on the screen

