# -*- coding: utf-8 -*-
"""
Created on Sat May 29 14:19:25 2021

@author: Selim Arslan
RungeKutta-Verfahren für DGL k-ter Ordnung
"""

import numpy as np
import matplotlib.pyplot as plt

#%% 4-Stufiges RungaKuttaVerfahren
def f(x,y):
    resultat = np.empty(y.shape)
    resultat[:-1] = y[1:]
    resultat[-1] = np.sin(x) + 5 - 1.1 * y[3] + 0.1 * y[2] + 0.3 * y[0]
    return resultat

def runge_kutta_4(f, x0, y0, xn, n):
    h = (xn - x0)/n
    x = np.linspace(x0, xn, n + 1)
    y = np.empty((n + 1, y0.size))
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(x[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        print('k1= ', k1)
        print('k2= ', k2)
        print('k3= ', k3)
        print('k4= ', k4)
        print('y[' + str(i + 1) + ']= ', y[i + 1])
        
    
    return x,y

   

x0 = 0
y0 = np.array([0.,2.,0.,0.])

x, y = runge_kutta_4(f, x0, y0, 1, 10)

for i in range(0, y.shape[1]):
    plt.plot(x,y[:,i])
        
plt.grid()
plt.legend(["Lösung y(x)", "y'(x)","y''(x)","y'''(x)"])

#%% EulerVerfahren für k-ter Ordnung

def f(x,y):
    resultat = np.empty(y.shape)
    resultat[:-1] = y[1:]
    resultat[-1] = np.sin(x) + 5 - 1.1 * y[3] + 0.1 * y[2] + 0.3 * y[0]
    return resultat


def eulerVerfahren(f, x0, y0, xn, n):
    h = (xn-x0) / n
    x = np.linspace(x0, xn, n+1)
    y = np.empty((n+1, y0.size))
    y[0] = y0
    
    for i in range(n):
        k1 = f(x[i], y[i])
        y[i + 1] = y[i] + h * k1
        
        print('k1 = ', k1)
        print('k[' + str(i+1) + '] =', y[i +1])
    return x,y

x0 = 0
y0 = np.array([0.,2.,0.,0.])

x, y = eulerVerfahren(f, x0, y0, 1, 10)

for i in range(0, y.shape[1]):
    plt.plot(x,y[:,i])
        
plt.grid()
plt.legend(["Lösung y(x)", "y'(x)","y''(x)","y'''(x)"])
#%% MittelPunktVerfahren für k-ter Ordnung
x0 = 0
v0 = 100
masse = 97000

a = 0
b = 20

h = 0.1

anzahlschritte = int((b-a)/h)
rows = 2

t = np.zeros(anzahlschritte+1)
z = np.zeros([rows, anzahlschritte+1])

z[0] = a
z[:,0] =np.array([x0,v0])

def f(t,z):
    resultat = np.empty(z.shape)
    resultat[:-1] = z[1:]
    resultat[-1] = (-5*z[1]**2 - 0.1 * z[0])/masse - 570000/masse
    return resultat

for i in range(0,anzahlschritte):
    t_half = t[i] + (h/2)
    z_half = z[:,i] + (h/2) * f(t[i], z[:,i]) 
    t[i+1] = t[i] + h
    z[:, i+1] = z[:,i] + h*f(t_half, z_half)
    
plt.plot(t,z[0,:],t,z[1,:]) 
plt.xlabel('Zeit [s]')
plt.ylabel('Weg [m] bzw, Geschw. [m/s]')
plt.legend(['Bremsweg', 'Geschwindigkeit'])
plt.grid()

"""
    ZUm Bremsen braucht es ca. 17 Sekunden und ca 800m wird er zum Stillstand kommen
"""

#%% Exakter Wert holen
ind = np.where(z[1,:] > 0) #Hole in der Zeile 1 Alle Werte die grösser als 0 sind
i = np.max(ind) #maximaler Wert holen
t_stop = t[i] # Stp Zeit holen 
v_stop = z[1,i] # Geschwindigkeit holen
x_stop = z[0,i] # und Ortspunklt holen
print('Stopzeit bei = ', t_stop)
print('Geschwindigkeit = ', v_stop)
print('Ortspunkt bei = ', x_stop)

