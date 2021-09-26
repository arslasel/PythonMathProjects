# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:27:40 2021

@author: Selim Arslan
Gauss-Newton und gedämptes Gauss-Newton-Verfahren
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.optimize
#%% GaussNewton

"""
Funktion f(x)= (lam0 + lam1*10**(lam2 + lam3*x)) / (1 + 10**(lam2 + lam3*x))
"""
x = sp.Matrix([2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. , 7.5, 8. ,
       8.5, 9. , 9.5])
y = sp.Matrix([159.57209984, 159.8851819 , 159.89378952, 160.30305273,
       160.84630757, 160.94703969, 161.56961845, 162.31468058,
       162.32140561, 162.88880047, 163.53234609, 163.85817086,
       163.55339958, 163.86393263, 163.90535931, 163.44385491])

abcd = sp.symbols('a b c d')

def f(x, lam):
    return (lam[0] + lam[1]*10**(lam[2]+lam[3]*x))/(1 + 10**(lam[2] + lam[3]*x))


#Erstellen der Matrix mit der Funktion und den Werten x, y
g = sp.Matrix([y[i]-f(x[i],abcd) for i in range(len(x))])
DG = g.jacobian(abcd)

#Umwandlung der Symbolischen Grössen g und Dg in numerische Funktion.
#Damit kann als Input ein Array akzeptiert werden
g = sp. lambdify([abcd], g, "numpy")
DG = sp.lambdify([abcd], DG, "numpy")
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
tol = 1e-5
nMax = 30
lam0 = np.array([100, 120, 3, -1],dtype=np.float64)
[lam1, n] = gauss_newton(g, DG, lam0, tol, nMax)
print(lam1, n)

#PLot
plt.figure(0)
x_new = np.arange(x[0], x[-1], 0.01)
plt.semilogy(x,y,'+', x_new, f(x_new,lam1))
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['data','fit'])
plt.show()

#%% gedämptes Gauss Newton

"""
Funktion f(x)= (lam0 + lam1*10**(lam2 + lam3*x)) / (1 + 10**(lam2 + lam3*x))
"""
x = sp.Matrix([2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. , 7.5, 8. ,
       8.5, 9. , 9.5])
y = sp.Matrix([159.57209984, 159.8851819 , 159.89378952, 160.30305273,
       160.84630757, 160.94703969, 161.56961845, 162.31468058,
       162.32140561, 162.88880047, 163.53234609, 163.85817086,
       163.55339958, 163.86393263, 163.90535931, 163.44385491])

abcd = sp.symbols('a b c d')

def f(x, lam):
    return (lam[0] + lam[1]*10**(lam[2]+lam[3]*x))/(1 + 10**(lam[2] + lam[3]*x))


#Erstellen der Matrix mit der Funktion und den Werten x, y
g = sp.Matrix([y[i]-f(x[i],abcd) for i in range(len(x))])
DG = g.jacobian(abcd)

#Umwandlung der Symbolischen Grössen g und Dg in numerische Funktion.
#Damit kann als Input ein Array akzeptiert werden
g = sp. lambdify([abcd], g, "numpy")
DG = sp.lambdify([abcd], DG, "numpy")

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
tol = 1e-5
nMax = 30
pMax = 5
damping = 1
lam0 = np.array([100, 120, 3, -1],dtype=np.float64)
[lam1, n] = gedämpftes_Gauss_Newton(g, DG, lam0, tol, nMax, pMax, damping)
print(lam1, n)

#PLot
plt.figure(1)
x_new = np.arange(x[0], x[-1], 0.01)
plt.semilogy(x,y,'+', x_new, f(x_new,lam1))
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['data','fit'])
plt.show()
#%% Analyse ob es konvergiert
print('\nUngedämptes Gauss-Newtwon-Verfahren:\n')
damping = 0
[lam2, n] = gedämpftes_Gauss_Newton(g, DG, lam0, tol, nMax, pMax, damping)
"""
Antwort: es divergiert
"""
#%% scypy optimize verwenden
print('\nscipy: \n')
def err_func(x):
    return np.linalg.norm(g(x))**2
xOptimize = scipy.optimize.fmin(err_func, lam0)
print(xOptimize)