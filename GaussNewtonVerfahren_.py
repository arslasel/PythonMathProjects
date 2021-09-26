# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:46:19 2021

@author: Selim Arslan
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
sp.init_printing()
#%%

x = sp.Matrix([25, 35, 45, 55, 65])
y = sp.Matrix([47, 114, 223, 81, 20])

sym = sp.symbols('A0 f0 c0')


def f(x, lam):
    return lam[0] / (((x**2 - lam[1]**2)**2) + lam[2]**2)


#Erstellen der Matrix mit der Funktion und den Werten x, y
g = sp.Matrix([y[i]-f(x[i],sym) for i in range(len(x))])
DG = g.jacobian(sym)

#Umwandlung der Symbolischen Grössen g und Dg in numerische Funktion.
#Damit kann als Input ein Array akzeptiert werden
g = sp. lambdify([sym], g, "numpy")
DG = sp.lambdify([sym], DG, "numpy")
def gauss_newton(g, Dg, lam0, tol, nMax):
    k=0
    lam=np.copy(lam0)
    increment = tol+1
    err_func = np.linalg.norm(g(lam))**2
    
    while increment>tol and k < nMax: #Hier kommt Ihre Abbruchbedingung, die tol und max_iter berücksichtigen muss# 

        # QR-Zerlegung von Dg(lam) und delta als Lösung des lin. Gleichungssystems
        [Q,R] = np.linalg.qr(Dg(lam))
        delta = np.linalg.solve(R, -Q.T @ g(lam)).flatten()  # Achtung: flatten() braucht es, um aus dem Spaltenvektor delta wieder
                                                             # eine "flachen" Vektor zu machen, da g hier nicht mit Spaltenvektoren als Input umgehen kann           
            
        # Update des Vektors Lambda        
        lam = lam + delta
        err_func = np.linalg.norm(g(lam),2)**2
        increment = np.linalg.norm(delta,2)
        k = k + 1
        print('Iteration: ',k)
        print('lambda = ',lam)
        print('Inkrement = ',increment)
        print('Fehlerfunktional =', err_func)
    return(lam,k)

print('\nGauss-Newton-Verfahren:\n')
tol = 10**-3
nMax = 20
lam0 = np.array([10**7, 35, 350],dtype=np.float64)
[lam1, n] = gauss_newton(g, DG, lam0, tol, nMax)
print(lam1, n)

#PLot
x_new = np.arange(25, 65, 1)
plt.plot(x, y, 'o')
plt.semilogy(x_new, f(x_new,lam1))
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['data','fit'])

#%% Gedämpftes GaussNewton



x = sp.Matrix([25, 35, 45, 55, 65])
y = sp.Matrix([47, 114, 223, 81, 20])

sym = sp.symbols('A0 f0 c0')


def f(x, lam):
    return lam[0] / (((x**2 - lam[1]**2)**2) + lam[2]**2)


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
tol = 10**-3
nMax = 20
pMax = 5
damping = 1
lam0 = np.array([10**7, 35, 350],dtype=np.float64)
[lam1, n] = gedämpftes_Gauss_Newton(g, DG, lam0, tol, nMax, pMax, damping)
print(lam1, n)

#PLot
x_new = np.arange(25, 65, 1)
plt.plot(x, y, 'o')
plt.semilogy(x_new, f(x_new,lam1))
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['data','fit'])

"""
c.) Das gedämpfte NewtonVerfahren konvergiert für einen grösseren Bereich vom 
    Startvektoren, darum geht es mit dem Gedämpftes Newtonverfahren
"""


"""
d.) Indem man das Maximum aus der Daten herrausnimt in diesem Fall ist es x = 45
    y = 223
"""






























































