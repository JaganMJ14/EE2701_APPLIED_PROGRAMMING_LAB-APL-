# Each section of the code produces either a plot or an output value.
# So, run each section of the code separately to get the desired answers for different sections of the assignment

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#1

import scipy.signal as sp
from matplotlib import pyplot as plt
import numpy as np
import pylab


num = np.poly1d([1, 0.5])
den = np.poly1d([1, 1, 2.5])
den = np.polymul([1, 0, 2.25], den)
H = sp.lti(num, den)
t, x = sp.impulse(H, None, np.linspace(0,50,500))
pylab.plot(t, x)
pylab.xlabel(r'$t\rightarrow$')
pylab.ylabel(r'$x(t)\rightarrow$')
pylab.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#2

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#3

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#4a

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#4b

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#5

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#6

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#7(for 10msec)

import scipy.signal as sp
from matplotlib import pyplot as plt
import numpy as np
import pylab

t = np.linspace(0,0.01,1000)
H = sp.lti(np.poly1d([1000000]),np.poly1d([0.000001,100,1000000]))
vi = np.cos(1000*t) - np.cos(1000000*t)
t, x, svec = sp.lsim(H, vi, t)
pylab.plot(t, x)
pylab.xlabel(r'$t\rightarrow$')
pylab.ylabel(r'$Vo(t)\rightarrow$')
pylab.grid()
pylab.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#EИD



