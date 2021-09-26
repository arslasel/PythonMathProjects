# -*- coding: utf-8 -*-
"""
Created on Sun May 30 22:06:55 2021

@author: Selim
"""

import numpy as np
import matplotlib.pyplot as plt

d_i = np.array([500.,  1000., 1500., 2500., 3500., 4000., 4500., 5000., 5250., 5500.])
P_i = np.array([10.5,  49.2,  72.1,  85.4,  113,   121,   112,   80.2,  61.1,  13.8])

plt.plot(d_i, P_i, '*')
plt.grid()
plt.show()

"""
Es muss ein Polynom 4. Grades sein.... so wies aussieht
Ansatz: P(x) = a4 * x**4 + a3*x**3 + a2*x**2 + a1*x + a0
"""

######## Aufgabe b)

dSkalierung = 1000
d = d_i/dSkalierung

def normalGleichung(x,y):
    
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
        return lamb[0] * f1(x) + lamb[1] * f2(x) + lamb[2]*f3(x) + lamb[3]*f4(x) + lamb[4] * f5(x)
    
    
    A = np.zeros([np.size(x),5]) ## 5 entspricht anzahl der funktionen (f1,f2,f3)
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
    
    x_fine = np.arange(500, 5500, 1.) #x[-1] ist das letzte Element und der Arange geht vom 1sten Elemnt bis zum letzten + 0.1 in 0.1 Schritten
    plt.plot(x,y,'o')
    plt.plot(x_fine,f(x_fine,lamb))
    plt.show()
    
    summ = 0
    for i in np.arange(np.size(x)):
        summ = summ + (y[i] - f(x[i], lamb))**2
        
    print('Fehlerfunktional')
    print(summ)
    
normalGleichung(d_i, P_i)  