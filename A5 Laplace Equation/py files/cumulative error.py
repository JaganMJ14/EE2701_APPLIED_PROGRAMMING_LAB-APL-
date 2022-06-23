import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import transpose
from numpy import exp


def error_fit(x, y):
    logy = np.log(y)
    itervec = np.zeros((len(x), 2))
    itervec[:, 0] = x
    itervec[:, 1] = 1
    B, logA = np.linalg.lstsq(itervec, np.transpose(logy))[0]
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
