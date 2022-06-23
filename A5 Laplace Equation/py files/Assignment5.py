# Each section of the code produces either a plot or an output value.
# So, run each section of the code separately to get the desired answers for different sections of the assignment

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Code 1
# Allocate and initialise the potential and plot it

import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np


Nx = 25
Ny = 25
radius = 8
Niter = 3000
phi = np.zeros((Ny, Nx), dtype=float)
x, y = np.linspace(-0.5, 0.5, 25, dtype=float), np.linspace(-0.5, 0.5, 25, dtype=float)
Y, X = np.meshgrid(y, x)
ii = np.where((X**2+Y**2)<(0.35**2))
phi[ii] = 1.0
pylab.figure(1)
pylab.plot(ii[0]/Nx-0.48, ii[1]/Ny-0.48, 'ro', label="V = 1")
pylab.contourf(X, Y, phi)
pylab.title("Initial Potential Contour")
pylab.xlim(-1, 1)
pylab.ylim(-1, 1)
pylab.xlabel(r'$X\rightarrow$')
pylab.ylabel(r'$Y\rightarrow$')
pylab.grid()
pylab.legend()
pylab.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Code 2
# Perform the iteration

import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np

Nx = 25
Ny = 25
radius = 8
Niter = 3000
phi = np.zeros((Ny, Nx), dtype=float)
x, y = np.linspace(-0.5, 0.5, 25, dtype=float), np.linspace(-0.5, 0.5, 25, dtype=float)
Y, X = np.meshgrid(y, x)
ii = np.where((X**2+Y**2)<(0.35**2))
phi[ii] = 1.0
err = np.zeros(Niter)
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1, 1:-1] = 0.25*(oldphi[1:-1, 0:-2] + oldphi[1:-1, 2:] + oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1])
    phi[:, 0] = phi[:, 1]
    phi[:, Nx-1] = phi[:, Nx-2]
    phi[0, :] = phi[1, :]
    phi[Ny-1, :] = 0
    phi[ii] = 1.0
    err[k] = np.max(np.abs(phi - oldphi))
    if err[k] == 0:
        print("Steady state reached at ",k," iterations !!!")
        break

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Error plot on loglog

import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np

Nx = 25
Ny = 25
radius = 8
Niter = 1500
phi = np.zeros((Ny, Nx), dtype=float)
x, y = np.linspace(-0.5, 0.5, 25, dtype=float), np.linspace(-0.5, 0.5, 25, dtype=float)
Y, X = np.meshgrid(y, x)
ii = np.where((X**2+Y**2)<(0.35**2))
phi[ii] = 1.0
err = np.zeros(Niter)
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1, 1:-1] = 0.25*(oldphi[1:-1, 0:-2] + oldphi[1:-1, 2:] + oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1])
    phi[:, 0] = phi[:, 1]
    phi[:, Nx-1] = phi[:, Nx-2]
    phi[0, :] = phi[1, :]
    phi[Ny-1, :] = 0
    phi[ii] = 1.0
    err[k] = np.max(np.abs(phi - oldphi))
    if err[k] == 0:
        break

pylab.loglog(range(Niter)[::50], err[::50], 'ro')
pylab.title("Error on a loglog plot")
pylab.xlabel("No of iterations")
pylab.ylabel("Error")
pylab.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Error Plot on semilog

import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np

Nx = 25
Ny = 25
radius = 8
Niter = 3000
phi = np.zeros((Ny, Nx), dtype=float)
x, y = np.linspace(-0.5, 0.5, 25, dtype=float), np.linspace(-0.5, 0.5, 25, dtype=float)
Y, X = np.meshgrid(y, x)
ii = np.where((X**2+Y**2)<(0.35**2))
phi[ii] = 1.0
err = np.zeros(Niter)
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1, 1:-1] = 0.25*(oldphi[1:-1, 0:-2] + oldphi[1:-1, 2:] + oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1])
    phi[:, 0] = phi[:, 1]
    phi[:, Nx-1] = phi[:, Nx-2]
    phi[0, :] = phi[1, :]
    phi[Ny-1, :] = 0
    phi[ii] = 1.0
    err[k] = np.max(np.abs(phi - oldphi))
    if err[k] == 0:
        break
pylab.semilogy(range(Niter)[::50], err[::50], 'ro')
pylab.title("Error on a semilog plot")
pylab.xlabel("No of iterations")
pylab.ylabel("Error")
pylab.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# 3 plots including the fit

import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp


def error_fit(x, y):
    logy = np.log(y)
    itervec = np.zeros((len(x), 2))
    itervec[:, 0] = x
    itervec[:, 1] = 1
    B, logA = np.linalg.lstsq(itervec, logy)[0]
    return (exp(logA), B)


def fit_exp(x, A, B):
    B = np.exp(B*x)
    A = np.A
    y = A*B
    return np.y

Nx = 25
Ny = 25
radius = 8
Niter = 1500
phi = np.zeros((Ny, Nx), dtype=float)
x, y = np.linspace(-0.5, 0.5, 25, dtype=float), np.linspace(-0.5, 0.5, 25, dtype=float)
Y, X = np.meshgrid(y, x)
ii = np.where((X**2+Y**2)<(0.35**2))
phi[ii] = 1.0
err = np.zeros(Niter)
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1, 1:-1] = 0.25*(oldphi[1:-1, 0:-2] + oldphi[1:-1, 2:] + oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1])
    phi[:, 0] = phi[:, 1]
    phi[:, Nx-1] = phi[:, Nx-2]
    phi[0, :] = phi[1, :]
    phi[Ny-1, :] = 0
    phi[ii] = 1.0
    err[k] = np.max(np.abs(phi - oldphi))
    if err[k] == 0:
        break
A, B = error_fit(range(Niter), err)
A_500, B_500 = error_fit(range(Niter)[500:], err[500:])
pylab.semilogy(range(Niter)[::50], err[::50], 'ro', label='original')
x = range(Niter)[::50]
y1 = A*np.exp(B*x)
y2 = A_500*np.exp(B_500*x)
pylab.semilogy(x, y1, 'go', label='fit1')
pylab.semilogy(x, y2, 'bo', label='fit2')
pylab.legend()
pylab.xlabel(r'Niter$\rightarrow$')
pylab.ylabel(r'Error$\rightarrow$')
pylab.title('Semilog plot of Error vs number of iterations')
pylab.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Stopping condition

import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp


def error_fit(x, y):
    logy = np.log(y)
    itervec = np.zeros((len(x), 2))
    itervec[:, 0] = x
    itervec[:, 1] = 1
    B, logA = np.linalg.lstsq(itervec, logy)[0]
    return (exp(logA), B)


def fit_exp(x, A, B):
    B = np.exp(B*x)
    A = np.A
    y = A*B
    return np.y

Nx = 25
Ny = 25
radius = 8
Niter = 1500
phi = np.zeros((Ny, Nx), dtype=float)
x, y = np.linspace(-0.5, 0.5, 25, dtype=float), np.linspace(-0.5, 0.5, 25, dtype=float)
Y, X = np.meshgrid(y, x)
ii = np.where((X**2+Y**2)<(0.35**2))
phi[ii] = 1.0
err = np.zeros(Niter)
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1, 1:-1] = 0.25*(oldphi[1:-1, 0:-2] + oldphi[1:-1, 2:] + oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1])
    phi[:, 0] = phi[:, 1]
    phi[:, Nx-1] = phi[:, Nx-2]
    phi[0, :] = phi[1, :]
    phi[Ny-1, :] = 0
    phi[ii] = 1.0
    err[k] = np.max(np.abs(phi - oldphi))
    if err[k] == 0:
        break
A, B = error_fit(range(Niter), err)
x = range(Niter)[::50]
X = np.arange(0.5, Niter + 0.5, 50)
y1 = -A/B*np.exp(B*X)
pylab.semilogy(x, y1, 'ro')
pylab.legend()
pylab.xlabel(r'Niter$\rightarrow$')
pylab.ylabel(r'Error$\rightarrow$')
pylab.title('Semilog plot of Cumulative Error')
pylab.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Surface plot of potential

import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp

Nx = 25
Ny = 25
radius = 8
Niter = 3000
phi = np.zeros((Ny, Nx), dtype=float)
x, y = np.linspace(-0.5, 0.5, 25, dtype=float), np.linspace(-0.5, 0.5, 25, dtype=float)
Y, X = np.meshgrid(y, x)
ii = np.where((X**2+Y**2)<(0.35**2))
phi[ii] = 1.0
err = np.zeros(Niter)
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1, 1:-1] = 0.25*(oldphi[1:-1, 0:-2] + oldphi[1:-1, 2:] + oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1])
    phi[:, 0] = phi[:, 1]
    phi[:, Nx-1] = phi[:, Nx-2]
    phi[0, :] = phi[1, :]
    phi[Ny-1, :] = 0
    phi[ii] = 1.0
    err[k] = np.max(np.abs(phi - oldphi))
    if err[k] == 0:
        break
fig1 = pylab.figure(4)
ax = p3.Axes3D(fig1)
pylab.title('The 3-D surface plot of the potential')
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=pylab.cm.jet)
pylab.xlabel(r'x$\rightarrow$')
pylab.ylabel(r'y$\rightarrow$')
ax.set_zlabel(r'$\phi\rightarrow$')
pylab.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Contour plot of potential

import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp

Nx = 25
Ny = 25
radius = 8
Niter = 3000
phi = np.zeros((Ny, Nx), dtype=float)
x, y = np.linspace(-0.5, 0.5, 25, dtype=float), np.linspace(-0.5, 0.5, 25, dtype=float)
Y, X = np.meshgrid(y, x)
ii = np.where((X**2+Y**2)<(0.35**2))
phi[ii] = 1.0
err = np.zeros(Niter)
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1, 1:-1] = 0.25*(oldphi[1:-1, 0:-2] + oldphi[1:-1, 2:] + oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1])
    phi[:, 0] = phi[:, 1]
    phi[:, Nx-1] = phi[:, Nx-2]
    phi[0, :] = phi[1, :]
    phi[Ny-1, :] = 0
    phi[ii] = 1.0
    err[k] = np.max(np.abs(phi - oldphi))
    if err[k] == 0:
        break
pylab.title("2D Contour plot of potential")
pylab.xlabel("X")
pylab.ylabel("Y")
pylab.plot(ii[0]/Nx-0.48, ii[1]/Ny-0.48, 'ro')
pylab.contourf(Y, X[::-1], phi)
pylab.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import transpose
from numpy import exp

Nx = 25
Ny = 25
radius = 8
Niter = 3000
phi = np.zeros((Ny, Nx), dtype=float)
x, y = np.linspace(-0.5, 0.5, 25, dtype=float), np.linspace(-0.5, 0.5, 25, dtype=float)
Y, X = np.meshgrid(y, x)
ii = np.where((X**2+Y**2)<(0.35**2))
phi[ii] = 1.0
err = np.zeros(Niter)
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1, 1:-1] = 0.25*(oldphi[1:-1, 0:-2] + oldphi[1:-1, 2:] + oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1])
    phi[:, 0] = phi[:, 1]
    phi[:, Nx-1] = phi[:, Nx-2]
    phi[0, :] = phi[1, :]
    phi[Ny-1, :] = 0
    phi[ii] = 1.0
    err[k] = np.max(np.abs(phi - oldphi))
    if err[k] == 0:
        break
Jx, Jy = (1/2*(phi[1:-1, 0:-2]-phi[1:-1, 2:]), 1/2*(phi[:-2, 1:-1]-phi[2:, 1:-1]))
pylab.title("Vector plot of current flow")
plt.quiver(Y[1:-1, 1:-1], X[1:-1, 1:-1], Jx[::-1,:], -Jy[::-1,:])
pylab.plot(ii[0]/Nx-0.48, ii[1]/Ny-0.48, 'ro')
pylab.show()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

