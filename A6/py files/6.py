import scipy.signal as sp
from matplotlib import pyplot as plt
import numpy as np
import pylab

t = np.linspace(0,30*0.000001,1000)
H = sp.lti(np.poly1d([1000000]),np.poly1d([0.000001,100,1000000]))
vi = np.cos(1000*t) - np.cos(1000000*t)
t, x, svec = sp.lsim(H, vi, t)
pylab.plot(t, x)
pylab.xlabel(r'$t\rightarrow$')
pylab.ylabel(r'$Vo(t)\rightarrow$')
pylab.grid()
pylab.show()