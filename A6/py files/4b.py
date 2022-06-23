import scipy.signal as sp
from matplotlib import pyplot as plt
import numpy as np
import pylab

t = np.linspace(0, 20, 1000)
num = np.poly1d([2])
den = np.poly1d([1, 0, 3, 0])
Hy = sp.lti(num, den)
t, y = sp.impulse(Hy, None, t)
pylab.plot(t, y)
pylab.xlabel(r'$t\rightarrow$')
pylab.ylabel(r'$y(t)\rightarrow$')
pylab.show()