# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:35:08 2021

@author: Selim Arslan
"""

import numpy as np
import matplotlib.pyplot as plt

#%% SEP2016 Aufgabe 5

# Daten
# =====

d_i = np.array([500.,  1000., 1500., 2500., 3500., 4000., 4500., 5000., 5250., 5500.])
P_i = np.array([10.5,  49.2,  72.1,  85.4,  113,   121,   112,   80.2,  61.1,  13.8])

# Aufgabe a 
plt.plot(d_i, P_i, 'o')
plt.grid()

"""
Sieht nach einem Polynom Grad 4 aus. 
Ansatz P(d) = a4*d^4 + a3*d^3 + a2*d^2  + a1*d + a0
"""


#Aufgabe b
# ----------
# Umskalierung der Drehzahlen: Teile die Drehzahlen beispielsweise durch 1000.

dscal = 1000.
d = d_i/dscal
# Normalgleichungssystem
n = len(d)
A = np.array([d**4, d**3, d**2, d, np.ones(n)]).T #Matrix A definieren
B = A.T@A 
c = A.T@P_i                                       

# Ausgleichspolynom
P = np.linalg.solve(A.T @ A, A.T @ P_i)     #Normalgleichung schon in solve                     
# Grafik
d_new = np.arange(500,5500,1)
Pv = np.polyval(P, d_new/dscal)
plt.plot(d_new, Pv) 

# Aufgabe c)
# ----------
"""
Mit Lehrer anschauen
"""

# Es muss die Nullstelle der 1. Ableitung dP(d) gefunden werden 
dP = np.polyder(P)                                # (0.5 Punkt)
#oder dP = np.array([4*P[0], 3*P[1], 2*P[2], P[3]])

# Zweite Ableitung ddP(d)                             # (0.5 Punkt)
ddP = np.polyder(dP)
# oder ddP = np.array([3*dP[0], 2*dP[1], dP[2]])

# Anfangswert
d = 4000./dscal                                       # (1 Punkt)

# Newton
while abs(np.polyval(dP, d)) > 1e-2:
    d = d - np.polyval(dP, d)/np.polyval(ddP, d)

dmax = d*dscal                                         # (1 Punkt)
print(dmax)

# Punkt in Grafik einzeichnen (nicht verlangt)
plt.plot(dmax, np.polyval(P, d), 'bx');

#%% Aufgabe 2b
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

#%%
di = np.array([500.,  1000., 1500., 2500., 3500., 4000., 4500., 5000., 5250., 5500.])
Pi = np.array([10.5,  49.2,  72.1,  85.4,  113,   121,   112,   80.2,  61.1,  13.8])

skalieren = 1000.
d = di / skalieren


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
        return lamb[0] * f1(x) + lamb[1] * f2(x) + lamb[2]*f3(x) + lamb[3] * f4(x) + lamb[4] * f5(x)
    
    
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
    
    x_fine = np.arange(500,5500,1.) #x[-1] ist das letzte Element und der Arange geht vom 1sten Elemnt bis zum letzten + 0.1 in 0.1 Schritten
    plt.plot(x,y,'o')
    plt.plot(x_fine,f(x_fine,lamb))
    plt.grid()
    
    
    summ = 0
    for i in np.arange(np.size(x)):
        summ = summ + (y[i] - f(x[i], lamb))**2
        
    print('Fehlerfunktional')
    print(summ)
    
normalGleichung(di, Pi)
