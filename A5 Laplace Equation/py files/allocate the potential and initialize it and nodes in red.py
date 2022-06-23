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
