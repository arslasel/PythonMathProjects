# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:01:35 2021

@author: Selim Arslan
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
sp.init_printing()


#%% SEP Aufgabe 6
#Daten
t_i = np.array([0.,  14.,   28.,    42.,    56.])
N_i = np.array([29., 2072., 15798., 25854., 28997.])

# Aufgabe a)
# ----------
# N(0) = N_0
# N(inf) = G

#%%                             
# Aufgabe b
plt.plot(t_i, N_i, 'o')
plt.grid()
# Aus dem plot ist ersichtlich, dass man nach 56 Tagen schon nahe am
# stationären Zustand N(inf) ist. Daher Also:

# Schätzung G=30000 (mit grosser Toleranz) 
# Schätzung N_0=29
#%%
# Aufgabe c
"""
N(t) = N_0 * np.exp(c*t)
umwandeln
ln(N(t)) = ln(N_0 * np.exp(c*t))
ln(N(t)) = c*t * ln(N_0 * np.exp())
ln(N(t)) = c*t * ln(N_0) + ln(np.exp())
ln(N(t)) = c*t * ln(N_0) + 1
c = ln(N(t)) / ln()
Umwandeln zu c = ln(N(t)) / ln(N0)
"""

c = (np.log(N_i[1]) - np.log(N_i[0]))/(t_i[1] - t_i[0])
print(c)



#%% Aufgabed

p = sp.symbols('G N0 c')

def N(t,p):
    return p[0]/((p[0]-p[1])/p[1]*sp.exp(-p[2]*t)+1)

g = sg = sp.Matrix([N_i[k]-N(t_i[k],p) for k in range(len(t_i))])
print(g)
Dg = g.jacobian(p)
print(Dg)

#Umwandlung der Symbolischen Grössen g und Dg in numerische Funktion.
#Damit kann als Input ein Array akzeptiert werden
g = sp.lambdify([p], g, 'numpy')
Dg = sp.lambdify([p], Dg, 'numpy')

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

print('\nGauss-Newton-Verfahren:\n')
tol = 1e-5
nMax = 30
lam0 = np.array([30000., 29., 0.305],dtype=np.float64)
pMax = 10
damping = 1
[lam1, n] = gedämpftes_Gauss_Newton(g, Dg, lam0, tol, nMax, pMax, damping)
print(lam1, n)

#Plot
t = sp.symbols('t')
F = N(t,lam1)
F = sp.lambdify([t], F, 'numpy')

t = np.arange(0.,56.,1)

plt.plot(t_i, N_i,'o', t, F(t))
plt.grid()

 

