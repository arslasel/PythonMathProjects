# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 08:31:16 2021

@author: Selim Arslan
"""


import numpy as np
import matplotlib.pyplot as plt

vRel = 2600
mA = 300000
mE = 80000
tE = 190
g = 9.81
u = (mA - mE) / tE


def f(x,y):
    resultat = np.empty(y.shape)
    resultat[:-1] = y[1:]
    resultat[-1] = vRel * (u / (mA - u * x)) - g - ((np.exp(-y[0] / 8000)) / (mA - u * x)) * y[1]
    return resultat

def runge_kutta_4(f, x0, y0, xn, n):
    h = (xn - x0)/n
    x = np.linspace(x0, xn, n + 1)
    y = np.empty((n + 1, y0.size))
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(x[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        print('k1= ', k1)
        print('k2= ', k2)
        print('k3= ', k3)
        print('k4= ', k4)
        print('y[' + str(i + 1) + ']= ', y[i + 1])
        
    
    return x,y

   

x0 = 0
y0 = np.array([0., 0.])

x, y = runge_kutta_4(f, x0, y0, tE, 10)


for i in range(0, y.shape[1]):
    plt.plot(x,y[:,i])

        
plt.grid()
plt.legend(["Lösung y(x)", "y'(x)","y''(x)","y'''(x)"])
