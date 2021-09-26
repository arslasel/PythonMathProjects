# -*- coding: utf-8 -*-
"""
Created on Mon May 31 10:41:51 2021

@author: Selim Arslan
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
sp.init_printing()

a = 2
b = 4

x, y = sp.symbols('x y')
f1 = y**2 + x**2 - 1
f2 = ((x - 2)**2 / a) + ((y - 1)**2 / b) - 1

f = sp.Matrix([f1, f2])

X = sp.Matrix([x, y])
Df = f.jacobian(X)

f = sp.lambdify([[[x],[y]]], f, "numpy") #Berechnung an verschiedenen Stellen
Df = sp.lambdify([[[x],[y]]], Df, "numpy") #Berechnung an verschiedenen Stellen

def newton(x, imax):
    i = 0
    residium = np.linalg.norm(f(x), 2)
    while residium > 10**-8 and i < imax:
        delta = np.linalg.solve(Df(x), -f(x))
        x = x + delta
        residium = np.linalg.norm(f(x), 2)
        i = i + 1
        print('Iteration: ', i)
        print('x= ', x)
        print('Residuum= ', residium)
        print()
    return x
        

print('Nullstelle1 = ')
Nullstelle1 = np.array([[2],[-1]])
x = newton(Nullstelle1, 10)

"""
Plot Funktion 1 und Funktion 2 
"""
p1 = sp.plot_implicit(sp.Eq(f1, 0))
p2 = sp.plot_implicit(sp.Eq(f2, 0))

p1.append(p2[0])
p1.show()



