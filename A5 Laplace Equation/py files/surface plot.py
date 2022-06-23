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