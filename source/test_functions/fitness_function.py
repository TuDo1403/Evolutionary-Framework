import numpy as np
from numpy import sin
from numpy import sqrt
from numpy import exp
from numpy import pi
from numpy import cos

def onemax(ind):
    f = sum(ind)
    return f

# Only use for maximize
def trap(ind, k):
    f = 0
    for i in range(0, len(ind), k):
        u = np.sum(ind[i : i + k])
        f += u if u == k else k - u - 1
    return f

# Only use for maximize
def trap_five(ind):
    return trap(ind, 5)

def cross_in_tray(ind):
    f = -0.0001 * (abs(sin(ind[0]) * sin(ind[1]) * exp(abs(100 - sqrt(ind[0]**2 + ind[1]**2)/pi))) + 1)**0.1
    return f

def himmelblau(ind):
    f = (ind[0]**2 + ind[1] - 11)**2 + (ind[0] + ind[1]**2 - 7)**2
    return f

def booth(ind):
    f = (ind[0] + 2*ind[1] - 7)**2 + (2*ind[0] + ind[1] - 5)**2
    return f

def rastrigin(ind):
    A = 10
    n = len(ind)
    f = A*n + np.sum(ind**2 - A*cos(2*pi*ind), axis=0)
    return f

def beale(ind):
    f = (1.5 - ind[0] + ind[0]*ind[1])**2 + (2.25 - ind[0] + ind[0]*(ind[1]**2))**2 + (2.625 - ind[0] + ind[0]*(ind[1]**3))**2
    return f

def happy_cat(ind):
    pass

# print(trap_five(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0])))

