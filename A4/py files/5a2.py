import pylab
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import exp
from numpy import cos
from numpy import sin

pie = math.pi
x1 = np.linspace(-2*pie, 4*pie, 300)
n = 51
B = np.zeros(n)
y1 = lambda x: exp(x)
B[0] = quad(y1, 0, 2*pie)[0]/(2*pie)
u = lambda x, k: exp(x)*np.cos(k*x)
v = lambda x, k: exp(x)*np.sin(k*x)
for i in range(1, n, 2):
    B[i] = quad(u, 0, 2*pie, args=((i+1)/2))[0]/pie
for i in range(2, n, 2):
    B[i] = quad(v, 0, 2*pie, args=(i/2))[0]/pie
x2 = np.linspace(0, 50, 51)
pylab.loglog(x2, np.abs(B), 'ro', label='Exact value')
pylab.xlabel(r'n$\rightarrow$')
pylab.ylabel(r'|Coefficient|$\rightarrow$')
pylab.grid()
pylab.legend()
x = np.linspace(0, 2*pie, 401)
x = x[:-1]
A = np.zeros((400, 51))
A[:, 0] = 1
for i in range(1,  26):
    A[:, 2*i-1] = cos(i*x)
    A[:, 2*i] = sin(i*x)
b = np.exp(x)
c = np.linalg.lstsq(A, b)[0]
pylab.loglog(x2, np.abs(c), 'go', label='least squares')
pylab.show()