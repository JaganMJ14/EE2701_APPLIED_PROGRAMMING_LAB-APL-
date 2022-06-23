import pylab
from scipy import linalg
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import exp
from numpy import cos
from numpy import sin

pie = math.pi
x = np.linspace(0, 2*pie, 401)
x = x[:-1]
A = np.zeros((400, 51))
A[:, 0] = 1
for i in range(1,  26):
    A[:, 2*i-1] = cos(i*x)
    A[:, 2*i] = sin(i*x)
b = np.cos(cos(x))
c = np.linalg.lstsq(A, b)[0]
approx = np.matmul(A, c)
pylab.plot(x, approx, 'go', label="Appoximation")
pylab.plot(x, b, 'r', label='Exact')
pylab.xlabel(r'n$\rightarrow$')
pylab.ylabel(r'$cos(cos(x))\rightarrow$')
pylab.grid()
pylab.legend()
pylab.show()
