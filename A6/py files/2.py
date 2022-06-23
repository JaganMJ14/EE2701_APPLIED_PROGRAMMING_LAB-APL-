import scipy.signal as sp
from matplotlib import pyplot as plt
import numpy as np
import pylab


num = np.poly1d([1, 0.05])
den = np.poly1d([1, 0.1, 2.2525])
den = np.polymul([1, 0, 2.25], den)
H = sp.lti(num, den)
t, x = sp.impulse(H, None, np.linspace(0,50,500))
pylab.plot(t, x)
pylab.xlabel(r'$t\rightarrow$')
pylab.ylabel(r'$x(t)\rightarrow$')
pylab.show()



