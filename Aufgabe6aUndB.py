# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:21:50 2021

@author: Selim Arslan
"""

import numpy as np
import sympy as sp0
import matplotlib.pyplot as plt
import scipy.optimize

#%% Aufgabe 6a
xi = np.array([0, 1, 2, 3, 4, 5])
yi = np.array([0.54, 0.44, 0.28, 0.18, 0.12, 0.08])

plt.plot(xi, yi, '*')
plt.grid()
plt.show()

#%% Aufgabe 6a
def normalGleichung(x,y):
    #Funktion x^2 + x + 1 (Aufgabenabhängig)
    def f1(x):
        return x**4
    
    def f2(x):
        return x**3
    
    def f3(x):
        return x**2
    
    def f4(x):
        return x
    
    def f5(x):
        return np.ones(x.shape)
    
    
    #Zusammensetzen aus f1,f2,f3
    def f(x,lamb):
        return lamb[0] * f1(x) + lamb[1] * f2(x) + lamb[2]*f3(x) + lamb[3]*f4(x) + lamb[4]*f5(x)
    
    
    A = np.zeros([np.size(x),5]) ## 3 entspricht anzahl der funktionen (f1,f2,f3)
    #Zusammensetzen aus f1,f2,f3
    A[:,0] = f1(x)
    A[:,1] = f2(x)
    A[:,2] = f3(x)
    A[:,3] = f4(x)
    A[:,4] = f5(x)
    
    
    [Q,R] = np.linalg.qr(A)
    lamb = np.linalg.solve(R, Q.T @ y)
    
    print("----------------------------------------")
    print("---------- Dies ist mit QR -------------")
    print("----------------------------------------")
    print("Dies ist lambdaQR:")
    print(lamb)
    print()
    print("Aufgabe 1b: Condition R (QR Verfahren)")
    print(np.linalg.cond(R))
    print()
    
    x_fine = np.arange(x[0],x[-1] + 0.1,0.1) #x[-1] ist das letzte Element und der Arange geht vom 1sten Elemnt bis zum letzten + 0.1 in 0.1 Schritten
    plt.plot(x,y,'o')
    plt.plot(x_fine,f(x_fine,lamb), color="blue")
    
    
    summ = 0
    for i in np.arange(np.size(x)):
        summ = summ + (y[i] - f(x[i], lamb))**2
        
    print('Fehlerfunktional')
    print(summ)
    
normalGleichung(xi, yi)

#%% Aufgabe 6b
def normalGleichung(x,y):
    #Funktion x^2 + x + 1 (Aufgabenabhängig)
    def f1(x):
        return x**2
    
    def f2(x):
        return np.ones(x.shape)
    
    
    #Zusammensetzen aus f1,f2,f3
    def f(x,lamb):
        return lamb[0] * f2(x)+ lamb[1] * f1(x) 
    
    
    A = np.zeros([np.size(x),2]) ## 3 entspricht anzahl der funktionen (f1,f2,f3)
    #Zusammensetzen aus f1,f2,f3
    A[:,0] = f1(x)
    A[:,1] = f2(x)

    
    
    [Q,R] = np.linalg.qr(A)
    lamb = np.linalg.solve(R, Q.T @ y)
    
    print("----------------------------------------")
    print("---------- Dies ist mit QR -------------")
    print("----------------------------------------")
    print("Dies ist lambdaQR:")
    print(lamb)
    print()
    print("Aufgabe 1b: Condition R (QR Verfahren)")
    print(np.linalg.cond(R))
    print()
    
    x_fine = np.arange(x[0],x[-1] + 0.1,0.1) #x[-1] ist das letzte Element und der Arange geht vom 1sten Elemnt bis zum letzten + 0.1 in 0.1 Schritten
    plt.plot(x,y,'o')
    plt.plot(x_fine,f(x_fine,lamb), color="blue")
    
    
    summ = 0
    for i in np.arange(np.size(x)):
        summ = summ + (y[i] - f(x[i], lamb))**2
        
    print('Fehlerfunktional')
    print(summ)
    
normalGleichung(xi, yi)
#%%


