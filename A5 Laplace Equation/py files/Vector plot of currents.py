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
