
from pylab import *
import scipy.special as sp
N=101                           
k=9                            

# generate the data points and add noise
t=linspace(0,10,N)              
y=1.05*sp.jn(2,t)-0.105*t       
Y=meshgrid(y,ones(k),indexing='ij')[0] 
scl=logspace(-1,-3,k)           
n=dot(randn(N,k),diag(scl))     
yy=Y+n                          


plot(t,yy)
xlabel(r'$t$',size=20)
ylabel(r'$f(t)+n$',size=20)
title(r'Plot of the data to be fitted')
grid(True)
savetxt("fitting.dat",c_[t,yy]) # write out matrix to file
show()
