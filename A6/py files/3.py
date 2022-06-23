import scipy.signal as sp
from matplotlib import pyplot as plt
import numpy as np
import pylab


num = np.poly1d([1])
den = np.poly1d([1, 0, 2.25])
H = sp.lti(num, den)
for omega in np.arange(1.4, 1.6, 0.05):
    t = np.linspace(0, 50, 500)
    u = np.cos(omega*t)*np.exp(-0.05*t)
    t, x, svec = sp.lsim(H, u, t)
    pylab.plot(t, x, label = 'w = ' + str(omega))
    pylab.xlabel(r'$t\rightarrow$')
    pylab.ylabel(r'$x(t)\rightarrow$')
    pylab.legend()
    pylab.show()



