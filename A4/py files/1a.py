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
