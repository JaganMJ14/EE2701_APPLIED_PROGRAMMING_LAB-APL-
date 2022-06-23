# Code 1
# Allocate and initialise the potential and plot it

import pylab
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import matplotlib.pyplot as plt
import numpy as np


import numpy as np

A = 1  # Area

Na = 5 * pow(10, 15)  # Doping concentration

q = 1.602 * pow(10, -19)  # Electron charge

ni = 1 * pow(10, 10) # /cm^3

µn = 1350

Vt = 0.026

Isc = 40 * pow(10, -3) # := 40 x 10^(-3)

τn = 1 * pow(10, -6)

Dn = µn * Vt
Ln = pow(Dn * τn, 0.5)

Io = (A * q * ni * ni * Dn) / (Ln * Na)
Il = Isc

Voc = Vt * np.log(1 + (Il / Io))


print("Voc = ", Voc * 1000, "mV")