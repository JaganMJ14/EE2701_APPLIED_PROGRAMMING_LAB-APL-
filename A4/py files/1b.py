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