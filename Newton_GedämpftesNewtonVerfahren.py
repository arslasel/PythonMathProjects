# -*- coding: utf-8 -*-
"""
Created on Thu May 27 09:31:36 2021

@author: Selim Arslan
"""
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
#%% Aufgabe 2a
x, y = sp.symbols('x y')
f1 = x**2 / 186**2 - y**2 / (300**2 - 186**2) - 1
f2 = (y-500)**2/279**2 - (x-300)**2 / (500**2 - 279**2) - 1

p1 = sp.plot_implicit(sp.Eq(f1, 0), (x, -2000, 2000), (y, -2000, 2000))
p2 = sp.plot_implicit(sp.Eq(f2, 0), (x, -2000, 2000), (y, -2000, 2000))

p1.append(p2[0])
p1.show()
#%% Aufgabe 2b

a, b, c = sp.symbols('a b c') #Symbole die Berechnet werden müssen
f1 = a + b * sp.exp(c * 1) - 40 #Alle Funktionen deklarieren auflösen = 0
f2 = a + b * sp.exp(c * 1.6) - 250
f3 = a + b * sp.exp(c * 2) - 800

f = sp.Matrix([f1, f2, f3]) # Alle Funktionen einfügen

X = sp.Matrix([a, b, c]) # Alle Symbole einfügen in die Matrix
Df = f.jacobian(X) 

print('Jacobian-Matrix= ', Df)
print('---------------------------\n')

print('Startverktor wird eingefügt: ')
print('---------------------------\n')
f0 = f.subs([(a, 1),(b, 2),(c, 3)]) # Startvektor explizit reinschreiben, nicht verwirren lassen
print('f0= ', f0)
print('---------------------------\n')
Df0 = Df.subs([(a, 1),(b, 2),(c, 3)])
print('Df0= ', Df0)
print('---------------------------\n')
print('Nummerischer Wert berechnen f0')
print(f0.evalf())
print('---------------------------\n')
print('Nummerischer Wert berechnen Df0')
print(Df0.evalf())
print('---------------------------\n')


f = sp.lambdify([[[a],[b],[c]]], f, "numpy") #Berechnung an verschiedenen Stellen, Alle Symbole von Zeile 22 einfügen.
Df = sp.lambdify([[[a],[b],[c]]], Df, "numpy") #Berechnung an verschiedenen Stellen

def newton(x, imax):
    i = 0
    residium = np.linalg.norm(f(x), 2)
    while residium > 10**-5 and i < imax:
        delta = np.linalg.solve(Df(x), -f(x))
        x = x + delta
        residium = np.linalg.norm(f(x), 2)
        i = i + 1
        print('Iteration: ', i)
        print('x= ', x)
        print('Residuum= ', residium)
        print()
        

print('Nullstelle1 = ')
Nullstelle1 = np.array([[1],[2],[3]]) # Startvektor eingeben
newton(Nullstelle1, 100) 


#%% Aufgab 3
"""
Allgemeine Version
"""

x1, x2, x3= sp.symbols('x1 x2 x3')

f1 = x1 + x2**2 - x3**2 - 13
f2 = sp.ln(x2/4) + sp.exp(0.5*x3-1) - 1
f3 = (x2 - 3)**2 - x3**3 + 7
f = sp.Matrix([f1, f2, f3])

x = sp.Matrix([x1, x2, x3])
Df = f.jacobian(x)
print('Jacobian-Matrix= ', Df)
print('---------------------------\n')

print('Startverktor wird eingefügt: ')
print('---------------------------\n')
f0 = f.subs([(x1, 0.996),(x2, 1.026)])
print('f0= ', f0)
print('---------------------------\n')
Df0 = Df.subs([(x1, 0.996),(x2, 1.026)])
print('Df0= ', Df0)
print('---------------------------\n')
print('Nummerischer Wert berechnen f0')
print(f0.evalf())
print('---------------------------\n')
print('Nummerischer Wert berechnen Df0')
print(Df0.evalf())
print('---------------------------\n')


f = sp.lambdify([[[x1],[x2],[x3]]], f, "numpy")
Df = sp.lambdify([[[x1],[x2],[x3]]], Df, "numpy")

def gedämpftes_Newton(f, Df, x0, tol, nMax, kMax):
    n = 0
    x = np.copy(x0)
    residium = np.linalg.norm(f(x), 2)
    
    while residium > tol and n < nMax:
        delta = np.linalg.solve(Df(x), -f(x))
        k = 0
        while np.linalg.norm(f(x + 0.5**k*delta), 2) >= residium and k <=kMax:
            k = k + 1
        if k == kMax + 1:
            k = 0
        x = x + 0.5**k*delta
        residium = np.linalg.norm(f(x), 2)
        n = n + 1
    return (x,n)

x0 = np.array([[1.5],[3],[2.5]])
tol = 10**-5
[xn, n] = gedämpftes_Newton(f, Df, x0, tol, 20, 4)
print('x_' + str(n) + ' = ' + str(np.reshape(xn,(3))))
        
#%% Newton-Verfahren
def f(x):
    return -5.39333 + 2.5236 * np.exp(2.8681 * x) - 1600


def df(x):
    return 7.23794 * np.exp(2.8681 * x)



def newton(x):
    return x - f(x)/df(x)

x0 = 2.

n = np.arange(6)
x = np.zeros(6)
x[0] = x0
for i in n[:-1]:
    x[i+1] = newton(x[i])
    print(x[i+1])



plt.figure()
plt.plot(n,x)
plt.grid()
plt.show()