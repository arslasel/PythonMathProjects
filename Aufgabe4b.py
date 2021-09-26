# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:52:35 2021

@author: Selim Arslan
"""
#%%Azfgabe4b
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x, y = sp.symbols('x y') #Symbole die Berechnet werden müssen
f1 = 25*x**2 / ((x**2 + 1) * (x + y)**2) #Alle Funktionen deklarieren auflösen = 0
f2 = 16*y**2 / ((y**2 + 1)*(x + y)**2)


f = sp.Matrix([f1, f2]) # Alle Funktionen einfügen

X = sp.Matrix([x,y]) # Alle Symbole einfügen in die Matrix
Df = f.jacobian(X) 

print('Jacobian-Matrix= ', Df)
print('---------------------------\n')

print('Startverktor wird eingefügt: ')
print('---------------------------\n')
f0 = f.subs([(x, 1),(y, 2)]) # Startvektor explizit reinschreiben, nicht verwirren lassen
print('f0= ', f0)
print('---------------------------\n')
Df0 = Df.subs([(x, 1),(y, 2)])
print('Df0= ', Df0)
print('---------------------------\n')
print('Nummerischer Wert berechnen f0')
print(f0.evalf())
print('---------------------------\n')
print('Nummerischer Wert berechnen Df0')
print(Df0.evalf())
print('---------------------------\n')


f = sp.lambdify([[[x],[y]]], f, "numpy") #Berechnung an verschiedenen Stellen, Alle Symbole von Zeile 22 einfügen.
Df = sp.lambdify([[[x],[y]]], Df, "numpy") #Berechnung an verschiedenen Stellen

def newton(x, imax):
    i = 0
    residium = np.linalg.norm(f(x), 2)
    while residium > 10**-5 and i < imax:
        delta = np.linalg.solve(Df(x), -f(x))
        x = x + delta
        residium = np.linalg.norm(f(x), 2)
        fehler = np.linalg.norm(x - delta)
        i = i + 1
        print('Iteration: ', i)
        print('x= ', x)
        print('Residuum= ', residium)
        print('fehler= ', fehler)
        print()
        

print('Nullstelle1 = ')
Nullstelle1 = np.array([[1],[2]]) # Startvektor eingeben
newton(Nullstelle1, 100) 