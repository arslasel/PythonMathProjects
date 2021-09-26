# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:39:33 2021

@author: Selim Arslan
"""

"""
Funktion f(x)= (lam0 + lam1*10**(lam2 + lam3*x)) / (1 + 10**(lam2 + lam3*x))


"""

import numpy as np
import sympy as sp0
import matplotlib.pyplot as plt
import scipy.optimize


T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
g = np.array([76, 92, 106, 123, 137, 151, 179, 203, 227, 250, 281, 309])

plt.plot(T, g, '*')
plt.grid()
plt.show()



def normalGleichung(x,y):
    #Funktion x^2 + x + 1 (Aufgabenabhängig)
    def f1(x):
        return x**3
    
    def f2(x):
        return x**2
    
    def f3(x):
        return x
    
    def f4(x):
        return np.ones(x.shape)
    
    
    #Zusammensetzen aus f1,f2,f3
    def f(x,lamb):
        return lamb[0] * f1(x) + lamb[1] * f2(x) + lamb[2]*f3(x) + lamb[3]*f4(x)
    
    
    A = np.zeros([np.size(x),4]) ## 3 entspricht anzahl der funktionen (f1,f2,f3)
    #Zusammensetzen aus f1,f2,f3
    A[:,0] = f1(x)
    A[:,1] = f2(x)
    A[:,2] = f3(x)
    A[:,3] = f4(x)
    
    
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
    
def normalGleichungZwei(x,y):
    
    def f1(x):
        return x**2
    
    def f2(x):
        return x
    
    def f3(x):
        return np.ones(x.shape)
    
    
    #Zusammensetzen aus f1,f2,f3
    def f(x,lamb):
        return lamb[0] * f1(x) + lamb[1] * f2(x) + lamb[2]*f3(x)
    
    
    A = np.zeros([np.size(x),3]) ## 3 entspricht anzahl der funktionen (f1,f2,f3)
    #Zusammensetzen aus f1,f2,f3
    A[:,0] = f1(x)
    A[:,1] = f2(x)
    A[:,2] = f3(x)
    
    
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
    plt.plot(x_fine,f(x_fine,lamb), '--', color="red")
    plt.show()
    
    summ = 0
    for i in np.arange(np.size(x)):
        summ = summ + (y[i] - f(x[i], lamb))**2
        
    print('Fehlerfunktional')
    print(summ)    
    
normalGleichung(T, g)
normalGleichungZwei(T, g)  