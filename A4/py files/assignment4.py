# Each section of the code produces either a plot or an output value.
# So, run each section of the code separately to get the desired answers for different sections of the assignment

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot of e^(x) and its periodic extension

import pylab
import numpy as np
import scipy.special as sp
import math

pie = math.pi
x1 = np.linspace(-2*pie, 4*pie, 300)
y1 = np.exp(x1)
x2 = np.linspace(0, 2*pie, 100)
tile1 = np.tile(x2, 3)
y2 = np.exp(tile1)
pylab.figure()
pylab.grid()
pylab.ylabel(r'$e^x\rightarrow$')
pylab.xlabel(r'x$\rightarrow$')
pylab.title(r'Figure 1')
pylab.semilogy(x1, y1, label='Actual plot')
pylab.semilogy(x1, y2, label='Periodic plot')
pylab.legend()
pylab.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot of coscos(x)

import pylab
import numpy as np
import scipy.special as sp
import math

pie = math.pi
x1 = np.linspace(-2*pie, 4*pie, 300)
y1 = np.cos(np.cos(x1))
pylab.plot(x1, y1)
pylab.grid()
pylab.xlabel(r'x$\rightarrow$')
pylab.ylabel(r'$\cos(cos(x))\rightarrow$')
pylab.title('Figure 2')
pylab.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Calculating the coefficient value for exp(x)

import pylab
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import exp

pie = math.pi
x1 = np.linspace(-2*pie, 4*pie, 300)
n = 51
A = np.zeros(n)
y1 = lambda x: exp(x)
A[0] = quad(y1, 0, 2*pie)[0]/(2*pie)
u = lambda x, k: exp(x)*np.cos(k*x)
v = lambda x, k: exp(x)*np.sin(k*x)
for i in range(1, n, 2):
    A[i] = quad(u, 0, 2*pie, args=((i+1)/2))[0]/pie
for i in range(2, n, 2):
    A[i] = quad(v, 0, 2*pie, args=(i/2))[0]/pie
for i in range(0, n):
    print(A[i])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Calculating coefficient value for coscos(x)

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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot between magnitude of coefficient with n for exp(x) in semilog

import pylab
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import exp

pie = math.pi
x1 = np.linspace(-2*pie, 4*pie, 300)
n = 51
A = np.zeros(n)
y1 = lambda x: np.exp(x)
A[0] = quad(y1, 0, 2*pie)[0]/(2*pie)
u = lambda x, k: exp(x)*np.cos(k*x)
v = lambda x, k: exp(x)*np.sin(k*x)
for i in range(1, n, 2):
    A[i] = quad(u, 0, 2*pie, args=((i+1)/2))[0]/pie
for i in range(2, n, 2):
    A[i] = quad(v, 0, 2*pie, args=(i/2))[0]/pie
x2 = np.linspace(0, 50, 51)
pylab.semilogy(x2, np.abs(A), 'ro')
pylab.xlabel(r'n$\rightarrow$')
pylab.ylabel(r'|Coefficient|$\rightarrow$')
pylab.grid()
pylab.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot between magnitude of coefficient with n for exp(x) in loglog

import pylab
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import exp

pie = math.pi
x1 = np.linspace(-2*pie, 4*pie, 300)
n = 51
A = np.zeros(n)
y1 = lambda x: np.exp(x)
A[0] = quad(y1, 0, 2*pie)[0]/(2*pie)
u = lambda x, k: exp(x)*np.cos(k*x)
v = lambda x, k: exp(x)*np.sin(k*x)
for i in range(1, n, 2):
    A[i] = quad(u, 0, 2*pie, args=((i+1)/2))[0]/pie
for i in range(2, n, 2):
    A[i] = quad(v, 0, 2*pie, args=(i/2))[0]/pie
x2 = np.linspace(0, 50, 51)
pylab.loglog(x2, np.abs(A), 'ro')
pylab.xlabel(r'n$\rightarrow$')
pylab.ylabel(r'|Coefficient|$\rightarrow$')
pylab.grid()
pylab.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot between magnitude of coefficient with n for coscos(x) in semilog

import pylab
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import cos

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
x2 = np.linspace(0, 50, 51)
pylab.semilogy(x2, np.abs(A), 'ro')
pylab.xlabel(r'n$\rightarrow$')
pylab.ylabel(r'|Coefficient|$\rightarrow$')
pylab.grid()
pylab.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot between magnitude of coefficient with n for coscos(x) in loglog

import pylab
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import cos

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
x2 = np.linspace(0, 50, 51)
pylab.loglog(x2, np.abs(A), 'ro')
pylab.xlabel(r'n$\rightarrow$')
pylab.ylabel(r'|Coefficient|$\rightarrow$')
pylab.grid()
pylab.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#least squares approach for exp(x)

from scipy import linalg
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import sin
from numpy import cos
from numpy import exp

pie = math.pi
x = np.linspace(0, 2*pie, 401)
x = x[:-1]
A = np.zeros((400, 51))
A[:, 0] = 1
for i in range(1, 26):
    A[:, 2*i-1] = cos(i*x)
    A[:, 2*i] = sin(i*x)
b = np.exp(x)
c = np.linalg.lstsq(A, b)[0]
print(c)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#least squares approach for coscos(x)

from scipy import linalg
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import sin
from numpy import cos
from numpy import exp

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
print(c)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot of magnitude of coefficients obtained from actual and least squares in semilog for exp(x)

import pylab
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import sin
from numpy import cos
from numpy import exp

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
pylab.grid()
pylab.legend()
pylab.semilogy(x2, np.abs(B), 'ro', label='Exact value')
pylab.xlabel(r'n$\rightarrow$')
pylab.ylabel(r'|Coefficient|$\rightarrow$')
x = np.linspace(0, 2*pie, 401)
x = x[:-1]
A = np.zeros((400, 51))
A[:, 0] = 1
for i in range(1,  26):
    A[:, 2*i-1] = cos(i*x)
    A[:, 2*i] = sin(i*x)
b = np.exp(x)
c = np.linalg.lstsq(A, b)[0]
pylab.semilogy(x2, np.abs(c), 'go', label='least squares')
pylab.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot of magnitude of coefficients obtained from actual and least squares in loglog for exp(x)

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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot of magnitude of coefficients obtained from actual and least squares in semilog for coscos(x)

import pylab
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import cos
from numpy import sin

pie = math.pi
x1 = np.linspace(-2*pie, 4*pie, 300)
n = 51
B = np.zeros(n)
y1 = lambda x: cos(cos(x))
B[0] = quad(y1, 0, 2*pie)[0]/(2*pie)
u = lambda x, k: cos(cos(x))*np.cos(k*x)
v = lambda x, k: cos(cos(x))*np.sin(k*x)
for i in range(1, n, 2):
    B[i] = quad(u, 0, 2*pie, args=((i+1)/2))[0]/pie
for i in range(2, n, 2):
    B[i] = quad(v, 0, 2*pie, args=(i/2))[0]/pie
x2 = np.linspace(0, 50, 51)
pylab.semilogy(x2, np.abs(B), 'ro', label='Exact value')
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
b = np.cos(cos(x))
c = np.linalg.lstsq(A, b)[0]
pylab.semilogy(x2, np.abs(c), 'go', label='least squares')
pylab.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot of magnitude of coefficients obtained from actual and least squares in loglog for coscos(x)

import pylab
from scipy.integrate import quad
import numpy as np
import scipy.special as sp
import math
from numpy import cos
from numpy import sin

pie = math.pi
x1 = np.linspace(-2*pie, 4*pie, 300)
n = 51
B = np.zeros(n)
y1 = lambda x: cos(cos(x))
B[0] = quad(y1, 0, 2*pie)[0]/(2*pie)
u = lambda x, k: cos(cos(x))*np.cos(k*x)
v = lambda x, k: cos(cos(x))*np.sin(k*x)
for i in range(1, n, 2):
    B[i] = quad(u, 0, 2*pie, args=((i+1)/2))[0]/pie
for i in range(2, n, 2):
    B[i] = quad(v, 0, 2*pie, args=(i/2))[0]/pie
x2 = np.linspace(1, 51, 51)
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
b = np.cos(cos(x))
c = np.linalg.lstsq(A, b)[0]
pylab.loglog(x2, np.abs(c), 'go', label='least squares')
pylab.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Maximum deviation in value of coefficients for exp(x) obtained by actual and least squares

from scipy import linalg
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
x = np.linspace(0, 2*pie, 401)
x = x[:-1]
A = np.zeros((400, 51))
A[:, 0] = 1
for i in range(1,  26):
    A[:, 2*i-1] = cos(i*x)
    A[:, 2*i] = sin(i*x)
b = np.exp(x)
c = linalg.lstsq(A, b)[0]
dev = abs(B - c)
maxdev = np.max(dev)
print(maxdev)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Maximum deviation in value of coefficients for coscos(x) obtained by actual and least squares

from scipy import linalg
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
y1 = lambda x: cos(cos(x))
B[0] = quad(y1, 0, 2*pie)[0]/(2*pie)
u = lambda x, k: cos(cos(x))*np.cos(k*x)
v = lambda x, k: cos(cos(x))*np.sin(k*x)
for i in range(1, n, 2):
    B[i] = quad(u, 0, 2*pie, args=((i+1)/2))[0]/pie
for i in range(2, n, 2):
    B[i] = quad(v, 0, 2*pie, args=(i/2))[0]/pie
x = np.linspace(0, 2*pie, 401)
x = x[:-1]
A = np.zeros((400, 51))
A[:, 0] = 1
for i in range(1,  26):
    A[:, 2*i-1] = cos(i*x)
    A[:, 2*i] = sin(i*x)
b = np.cos(cos(x))
c = linalg.lstsq(A, b)[0]
dev = abs(B - c)
maxdev = np.max(dev)
print(maxdev)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot of function exp(x) obtained by actual and least squares in semilog

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
b = np.exp(x)
c = np.linalg.lstsq(A, b)[0]
approx = np.matmul(A, c)
pylab.semilogy(x, approx, 'go', label="Appoximation")
pylab.semilogy(x, b, 'r', label='Exact')
pylab.xlabel(r'n$\rightarrow$')
pylab.ylabel(r'$e^{x}\rightarrow$')
pylab.grid()
pylab.legend()
pylab.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Plot of function coscos(x) obtained by actual and least squares

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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






