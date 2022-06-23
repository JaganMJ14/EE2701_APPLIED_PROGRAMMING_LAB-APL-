import scipy.signal as sp
from matplotlib import pyplot as plt
import numpy as np
import pylab

H = sp.lti(np.poly1d([1000000]),np.poly1d([0.000001,100,1000000]))
w, S, phi=H.bode()
pylab.subplot(2, 1, 1)
pylab.semilogx(w, S)
pylab.ylabel(r'$|H(s)|$')
pylab.xlabel(r'$w$')
pylab.subplot(2, 1, 2)
pylab.semilogx(w, phi)
pylab.ylabel(r'$\angle(H(s))$')
pylab.xlabel(r'$w$')
pylab.show()