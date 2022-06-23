from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
from matplotlib import cm
t = linspace(-pi,pi,1025); t = t[:-1]
dt = t[1]-t[0]; fmax = 1/dt
tarray = split(t,16)
Ymag = zeros((16,64))
Yphase = zeros((16,64))
for i in range(len(tarray)):
	n = arange(64)
	wnd = fftshift(0.54+0.46*cos(2*pi*n/64))
	y = cos(16*tarray[i]*(1.5 + tarray[i]/(2*pi)))*wnd
	y[0]=0
	y = fftshift(y)
	Y = fftshift(fft(y))/64.0
	Ymag[i] = abs(Y)
	Yphase[i] = angle(Y)
w = linspace(-fmax*pi,fmax*pi,64+1); w = w[:-1]
t=t[::64]
t,w = meshgrid(t,w)
fig = figure()
ax = p3.Axes3D(fig)
surface=ax.plot_surface(w,t,Yphase.T,cmap='viridis',linewidth=0, antialiased=False)
fig.colorbar(surface, shrink=0.5, aspect=5)
ax.set_title('surface plot');
ylabel(r"$\omega\rightarrow$")
xlabel(r"$t\rightarrow$")
show()