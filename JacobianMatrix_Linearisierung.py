# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:35:18 2021

@author: Selim Arslan
Serie 2  
"""

import numpy as np
import sympy as sp

#%% Aufgabe 2a
x1, x2 = sp.symbols('x1 x2')
f1 = 5 * x1 * x2
f2 = x1**2 * x2**2 + x1 + 2*x2

f = sp.Matrix([f1, f2])

X = sp.Matrix([x1, x2])
Df = f.jacobian(X)
print('Jacobian-Matrix= ', Df)
print('---------------------------')

print('Startverktor wird eingefügt: ')

Df0 = Df.subs([(x1, 1),(x2, 2)])
print(Df0)
print('---------------------------')
print('Nummerischer Wert berechnen')
print(Df0.evalf())
#%% Aufgabe 2b
x1, x2, x3 = sp.symbols('x1 x2 x3')

f1 = sp.ln(x1**2 + x2**2) + x3**2
f2 = sp.exp(x2**2 + x3**2) + x1**2
f3 = (1 / (x3**2 + x1**2)) + x2**2


f = sp.Matrix([f1, f2, f3])

X = sp.Matrix([x1, x2, x3])
Df = f.jacobian(X)
print('Jacobian-Matrix= ', Df)
print('---------------------------')

print('Startverkrot wird eingefügt')
Df0 = Df.subs([(x1,1),(x2,2),(x3,3)])
print(Df0)
print('---------------------------')
print('Nummerischer Wert berechnen')
print(Df0.evalf())
#%% Aufgabe 3
x1, x2, x3 = sp.symbols('x1 x2 x3')



f1 = x1 + x2**2 - x3**2 - 13
f2 = sp.ln(x2/4) + sp.exp(0.5*x3 - 1) - 1
f3 = (x2 - 3)**2 - x3**3 + 7

f = sp.Matrix([f1, f2, f3])
print('Funktionen in der Matrix= ', f)
print('-----------------------------')

f0 = f.subs([(x1, 1.5),(x2, 3),(x3, 2.5)])
print('Funktionen mit Stelle x(0)= ', f0)
print('-----------------------------')

X = sp.Matrix([x1, x2, x3])
Df = f.jacobian(X)
print('Jacobian-Matrix= ', Df)
print('---------------------------')

Df0 = Df.subs([(x1, 1.5),(x2, 3),(x3, 2.5)])
print('Jacobian-Matrix mit x(0)= ', Df0)
print('---------------------------')

T = sp.Matrix([x1 - 1.5, x2 - 3, x3 - 2.5])

g = f0 + Df0 * T
print('Linearisierung von-> g(x) = f0 + Df0 * T')
print(g)
print('---------------------------')
print('Nummerischer Wert berechnen')
print(g.evalf())

