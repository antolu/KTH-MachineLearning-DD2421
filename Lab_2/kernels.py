import numpy, math

def linear_kernel(x, y) :
    return numpy.dot(x, y) + 1

def polynomial_kernel(x, y, p=3) :
    return numpy.power(numpy.dot(x, y) + 1, p)

def radial_kernel(x, y, sigma=2) :
    diff = numpy.subtract(x, y)
    return math.exp((-numpy.dot(diff, diff)) / (2 * (sigma**2)))