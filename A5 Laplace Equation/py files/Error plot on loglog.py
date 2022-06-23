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