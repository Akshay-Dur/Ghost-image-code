# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:18:46 2023

@author: Akshay
"""

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from sympy.plotting import plot3d

#################################Taylor expansion#######################################
x=sym.Symbol('x')
#f=sym.diff(x**2) #Derivative
#print(f.subs(x, 1)) #Evaluate function f at x=1
'''The imaginary number is written as sym.I'''
f=sym.exp(sym.I*x) 
def taylor_exp(f, x, about, order): #Taylor expanding function f with respect to the variable x, around "about" 
    for i in range(order+1):
        if i==0:
            taylor_exp=f.subs(x, about)
            deriv=f
        else:
            deriv=sym.diff(deriv, x)
            taylor_exp+=deriv.subs(x, about)*(x-about)**i/sym.factorial(i)
    return taylor_exp
#taylor_exp(f, x, 0, 7).subs(x, 2.9) #Evaluate Taylor expanstion at x=2.9

#################################Ghost image calculation with Taylor expansion############
'''Initial conditions and definitions'''
X_0, X_0y=sym.symbols("X_0, X_0y", real=True)
#symbolsToDelete = ('X_0, X_0y') #To delete symbol status of necessary
# Position of aperture
X=0
Y=0
lmda=3 #Units are micrometers
n_e=(1+(1.151075*lmda**2)/(lmda**2-0.007142)+(0.21803*lmda**2)/(lmda**2-0.02259)+(0.656*lmda**2)/(lmda**2-263))**(0.5) 
n_0=(1+(0.90291*lmda**2)/(lmda**2-0.003926)+(0.83155*lmda**2)/(lmda**2-0.018786)+(0.76536*lmda**2)/(lmda**2-60.01))**(0.5) 
x=0*(180/np.pi)**(-1) #Set at 0 degrees
n_p=(n_e*n_0)/(n_0**2*(np.sin(x))**2+n_e**2*(np.cos(x))**2)**0.5
theta=np.arccos(n_p/n_0) #Radians
lmda=lmda*10**(-6)
omega_d=2*np.pi*((3*10**(8))/(lmda*n_0))
def k(n, omega_d, theta):
    d=((n*omega_d)/(3*10**(8)))*np.cos(theta)
    return d
chi=k(n_0, omega_d, 0)**2*(k(n_0, omega_d, theta))**(-1)-k(n_0, omega_d, theta)
w_0=0.26
w_p=10**(-1) #meters
L=0.001   #meters
#p=L/(2*w_p**2*k(n_0, omega_d, theta)) #Will treat as symbol temporarily due to taylor expansion
p=sym.Symbol('p')
t_1=sym.Symbol('t_1')
g=sym.I*(-w_p**2+2*sym.I*t_1*w_p**2*p)/(w_0**2+w_p**2+2*sym.I*t_1*w_p**2*p)
coeff=(1/(-(1/4)*(w_p**2)-sym.I*t_1*w_p**2*p/2+((1/16)*w_p**4-(sym.I*t_1*w_p**4*p)/(4)-(t_1**2*w_p**4*p**2)/(4))/((1/4)*w_0**2+(1/4)*w_p**2+(sym.I*w_p**2*t_1*p)/2)))*(1/((2*sym.pi)**3*((1/4)*w_0**2+(1/4)*w_p**2+(sym.I*w_p**2*t_1*p)/2))) #Ignoring numerator of coefficient in Eq (24) since this will cancel out with normalization factor (mu)
exponential=sym.exp(sym.I*chi*t_1*L-(X_0**2+X_0y**2)/(w_0**2+w_p**2+2*sym.I*t_1*w_p**2*p)-(-(X**2+Y**2)+2*sym.I*g*(X*X_0+Y*X_0y)+g**2*(X_0**2+X_0y**2))/(4*(-(1/4)*(w_p**2)-sym.I*t_1*w_p**2*p/2+((1/16)*w_p**4-(sym.I*t_1*w_p**4*p)/(4)-(t_1**2*w_p**4*p**2)/(4))/((1/4)*w_0**2+(1/4)*w_p**2+(sym.I*w_p**2*t_1*p)/2))))
'''Integrand Taylor expansion'''
integrand=coeff*exponential
Taylor_integrand=taylor_exp(integrand, p, 0, 1) #Taylor expansion here

'''Diamond product'''
#It is assumed that the transmission function is replaced with a dirac delta function. This means that we are calculating the impulse response function
#We ignore h(\omega) since this will be cancelled out by the normalization factor (mu)
p_r=L/(2*w_p**2*k(n_0, omega_d, theta))#We now define p
integral=sym.integrate(Taylor_integrand, (t_1, 0, 1)).subs(p, p_r)
MBEEBM=(sym.re(integral)**2+sym.im(integral)**2) #t^2(x)=delta() and overall factors are ignored due to normalization (mu)
mu=(1/4)*sym.integrate(MBEEBM, (X_0, -sym.oo, sym.oo), (X_0y, -sym.oo, sym.oo))
GI=MBEEBM/(4*mu)
#norm_check=sym.integrate(GI, (X_0, -sym.oo, sym.oo), (X_0y, -sym.oo, sym.oo)).subs(p, p_r)

#############################Plotting a 3D figure###################################

zoom=0.5
Fig=plot3d(GI, (X_0, -zoom, zoom), (X_0y, -zoom, zoom))



#Saving to computer
#Fig.save('Taylor Probability density') 
#plt.savefig('Taylor Probability density.png', dpi=300)
