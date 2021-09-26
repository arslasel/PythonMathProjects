# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 08:43:05 2021

@author: Selim Arslan
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.optimize


x = sp.Matrix([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = sp.Matrix([25.97, 43.64, 55.97, 61.12, 64.44, 68.42, 70.71, 70.82, 70.97, 70.39, 70.81])

plt.plot(x, y, 'o')

"""

Aufgabe 3a
Gemäss Plott hätte ich für A = 10, Q = 70 und t = 0.6 genommen
"""



#%% Aufgabe 3b

sym = sp.symbols('A Q t')

def f(x, lam):
    return lam[0] + (lam[1] - lam[0]) * (1 - sp.exp(-x/lam[2]))


#Erstellen der Matrix mit der Funktion und den Werten x, y
g = sp.Matrix([y[i]-f(x[i],sym) for i in range(len(x))])
DG = g.jacobian(sym)

#Umwandlung der Symbolischen Grössen g und Dg in numerische Funktion.
#Damit kann als Input ein Array akzeptiert werden
g = sp. lambdify([sym], g, "numpy")
DG = sp.lambdify([sym], DG, "numpy")

def gedämpftes_Gauss_Newton(g, DG, lam0, tol, nMax, pMax, damping):
    n = 0
    lam = np.copy(lam0)
    inkrement = tol + 1
    err_func = np.linalg.norm(g(lam))**2
    
    while inkrement > tol and n < nMax:
        #QR-Zerlegung
        [Q,R] = np.linalg.qr(DG(lam))
        delta = np.linalg.solve(R, -Q.T @ g(lam)).flatten()
        
        #dämpfung
        p = 0
        while damping and np.linalg.norm(g(lam+delta/(2.**p)))**2 >= err_func and p<=pMax:
            p = p + 1
        if p == pMax + 1:
            p = 0
        
        lam = lam + delta * (0.5**p)
        err_func = np.linalg.norm(g(lam))**2
        inkrement = np.linalg.norm(delta/(2.**p))
        n += 1
        print('Iteration: ', n)
        print('Lambda =  ', lam)
        print('Inkrement ', inkrement)
        print('Fehlerfunktion = ', err_func)
        
    return (lam, n)

#Aufgabe a)
print('\ngedämpftes Gauss-Newton-Verfahren:\n')
tol = 10**-7
nMax = 5
pMax = 5
damping = 1
lam0 = np.array([10., 80., 0.4],dtype=np.float64)
[lam1, n] = gedämpftes_Gauss_Newton(g, DG, lam0, tol, nMax, pMax, damping)
print(lam1, n)

#%% Aufgabe3c

def f(x):
    return 25.86997716 + (71.60605761 - 25.86997716) * (1 - np.exp(-x/0.19687068))

x = np.arange(0, 1.5, 0.001)
plt.plot(x, f(x))



#%% Aufgabe3d
"""
Ja ist wichtig, wenn der Startvektor zu weit ist konvergiert es nicht 
"""
