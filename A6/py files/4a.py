import scipy.signal as sp
from matplotlib import pyplot as plt
import numpy as np
import pylab

t = np.linspace(0, 20, 1000)
num = np.poly1d([1, 0, 2])
den = np.poly1d([1, 0, 3, 0])
Hx = sp.lti(num, den)
t, x = sp.impulse(Hx, None, t)
pylab.plot(t, x)
pylab.xlabel(r'$t\rightarrow$')
pylab.ylabel(r'$x(t)\rightarrow$')
pylab.show()