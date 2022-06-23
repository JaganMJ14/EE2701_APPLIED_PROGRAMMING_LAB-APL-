import pylab
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import exp
from numpy import coscos

pie = math.pi
x1 = np.linspace(-2*pie, 4*pie, 300)
n = 51
A = np.zeros(n)
y1 = lambda x: cos(cos(x))
A[0] = quad(y1, 0, 2*pie)[0]/(2*pie)
u = lambda x, k: cos(cos(x))*np.cos(k*x)
v = lambda x, k: cos(cos(x))*np.sin(k*x)
for i in range(1, n, 2):
    A[i] = quad(u, 0, 2*pie, args=((i+1)/2))[0]/pie
for i in range(2, n, 2):
    A[i] = quad(v, 0, 2*pie, args=(i/2))[0]/pie
for i in range(0, n):
    print(A[i])