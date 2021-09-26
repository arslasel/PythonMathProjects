# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:03:13 2021

@author: Selim Arslan
"""
import numpy as np
import matplotlib.pyplot as plt
#%% Aufgabe 5b
def f(t,y):
    return t / y

def y(t):
    return (t**2 - 3)**(1/2)

def rungeKuttaButcher(f, a, b, n, y0):
    h = (b-a) / n
    x = np.linspace(a, b, n+1)
    y = np.empty(n + 1)
    y[0] = y0
    
    for i in range(n):
        #Steigung berechnen
        #Steigungen berechnen
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (1/3) * h, y[i] + (1/3) * h * k1)
        k3 = f(x[i] + (2/3) * h, y[i] + (2/3) * h * k2)
        k4 = f(x[i] * h, y[i] + h * k3)
        print('k1= ', k1, 'i= ', i)
        print('k2= ', k2, 'i= ', i)
        print('k3= ', k3, 'i= ', i)
        print('k4= ', k4, 'i= ', i)
        #Y-Wert Berechnen
        y[i + 1] = y[i] + h * ((1/4)*k1 + k2 + (0.75)*k3 + k4)
        
        #Numerischen Wert anzeigen
        print('%.4f\t%.4f\t%.4f'% (a,b,y[i]))
        print('-------------------------')
    return x, y

#In ein Array speichern für die Verallgemeinerung
#Absoluter Fehler noch plotten für den Vergleich

x = np.linspace(1, 6)
#plot
a = 2
b = 5
h = 0.1
n = int((b-a)/h)
x_RungeKutta1, y_RungeKutta1 = rungeKuttaButcher(f, 2, 5, n, 5)
plt.plot(x_RungeKutta1, y_RungeKutta1)
plt.legend(['Runge-Kutta'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid('major')
plt.title('Runge-Kutta Verfahren')
plt.show()
#%% Aufgabe5c

err_10 = np.abs(y_RungeKutta1 - y(x_RungeKutta1)) 

print('err_10 = ', err_10[-1])


plt.figure()
plt.semilogy(x_RungeKutta1, err_10)
plt.grid()
plt.xlabel('err')
plt.ylabel('y')
plt.grid('major')
plt.title('Absoluter Fehler')
plt.legend(['Runga-Kutta'])
plt.show()
