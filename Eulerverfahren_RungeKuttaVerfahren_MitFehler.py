# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:00:53 2021

@author: Selim Arslan
"""

import numpy as np
import matplotlib.pyplot as plt

def f(t,y):
    return 0.1*y + np.sin(2*t)


def eulerVerfahren(f, a, b, n, y0):
    h = (b - a) / n
    y = [y0 for _ in range(n + 1)]
    for i in range(n):
        xi = a + i * h
        y[i + 1] = y[i] + h * f(xi, y[i])
    return y


def rungeKutta(f, a, b, n, y0):
    h = (b-a) / n
    x = np.linspace(a, b, n+1)
    y = np.empty(n+1)
    y[0] = y0
    
    for i in range(0,n):
        #Steigung berechnen
        k1 = f(x[i], y[i])
        k2 = f(x[i] + 0.5 * h, y[i] + 0.5 * h * k1)
        k3 = f(x[i] + 0.5 * h, y[i] + 0.5 * h * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        #Y-Wert Berechnen
        y[i + 1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        print('%.4f\t%.4f\t%.4f'% (a,b,y[i]))
        print('-------------------------')
    return x,y


a, b, n, y0 = 0, 6, 30, 0


x_small = np.linspace(a, b, 31)

x_RungeKutta, y_RungeKutta = rungeKutta(f, a, b, n, y0)
y_eulerWert = eulerVerfahren(f, a, b, n, y0)


plt.plot(x_small, y_eulerWert, '--', x_RungeKutta, y_RungeKutta)
#%% Fehlerberechnung


a, b, n, y0 = 0, 6, 30, 0


x_small = np.linspace(a, b, 31)

x_RungeKutta, y_RungeKutta = rungeKutta(f, a, b, n, y0)
y_eulerWert = eulerVerfahren(f, a, b, n, y0)


plt.semilogy(x_small, np.abs(y_eulerWert - y_RungeKutta), x_RungeKutta, np.abs(y_RungeKutta - y_eulerWert))