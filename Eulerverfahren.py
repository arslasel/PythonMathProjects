# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:52:10 2021

@author: Selim Arslan
Aufgabe 8.3
"""

import numpy as np
import matplotlib.pyplot as plt

def f(t,y):
    return t**2 + 0.1 * y

def sol(t):
    return -10*(t**2) - (200 * t) - 2000 + 1722.5 * np.exp(0.05 * (2*t + 3))

def euler(f, x0, y0, xn, n):
    h = (xn - x0)/ n
    x = np.linspace(x0, xn, n + 1)
    y = np.empty(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i],y[i])
    return x,y

k = 5
x, y = np.meshgrid(np.linspace(-1.5, 1.5, k), np.linspace(-1.5, 1.5, k))
n = np.sqrt(1 + f(x,y)**2)
u = 1/n
v = f(x,y)/n

plt.quiver(x, y, u, v, color='blue', width=0.04)

x_ = np.linspace(-1.5, 1.5)
plt.plot(x_, sol(x_), color='red')
plt.plot(x_,f(x_,x_), color='green')

plt.show()