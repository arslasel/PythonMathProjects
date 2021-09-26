# -*- coding: utf-8 -*-
"""
Created on Sun May 30 09:55:57 2021

@author: Selim Arslan
"""

import numpy as np
import matplotlib.pyplot as plt

#%% SEP  Aufgabe 2a
def f(t,y):
    return 1 - y/t

def y(t):
    return t/2. + 9./(2.*t)

def rungeKutta(f, a, b, n, y0):
    h = (b-a) / n
    x = np.linspace(a, b, n+1)
    y = np.empty(n+1)
    y[0] = y0
    
    for i in range(0,n):
        #Steigung berechnen
        k1 = f(x[i], y[i])
        k2 = f(x[i] + 0.5 * h, y[i] + 0.5 * h * k1)
        k3 = f(x[i] + 0.5 * h, y[i] + 0.5 * h * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        print('k1= ', k1, 'i= ', i)
        print('k2= ', k2, 'i= ', i)
        print('k3= ', k3, 'i= ', i)
        print('k4= ', k4, 'i= ', i)
        #Y-Wert Berechnen
        y[i + 1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        print('%.4f\t%.4f\t%.4f'% (a,b,y[i]))
        print('-------------------------')
    return x,y


x = np.linspace(1, 6)
#plot
plt.figure(1)
x_RungeKutta, y_RungeKutta = rungeKutta(f, 1, 6, 500, 5)
plt.plot(x, y(x), '-', x_RungeKutta, y_RungeKutta, '--,')
plt.legend(['Runge-Kutta'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid('major')
plt.title('Runge-Kutta Verfahren')

#%% SEP Aufgabe 2b
def f(t,y):
    return 1 - y/t

def y(t):
    return t/2. + 9./(2.*t)

def rungeKuttaButcher(f, a, b, n, y0):
    h = (b-a) / n
    x = np.linspace(a, b, n+1)
    t = np.zeros(n + 1)
    t[0] = a
    y = np.empty(n + 1)
    y[0] = y0
    
    for i in range(0,n):
        #Steigung berechnen
        #Steigungen berechnen
        k1 = f(t[i] + 0.25, y[i])
        k2 = f(t[i] + 0.5 * h, y[i] + h * 0.25 * k1)
        k3 = f(t[i] + 0.5 * h, y[i] + h * (0.5 * k1 + 0.5 * k2))
        k4 = f(t[i] + 0.75 * h, y[i] + h * (0.75 * k1 + 0.75 * k2 + 0.75 * k3))
        
        t[i + 1] = t[i] + h
        #Y-Wert Berechnen
        y[i + 1] = y[i] + h * (k1 + 4.*k2 + 4.*k3 + k4) / 10.
        
        #Numerischen Wert anzeigen
        print('%.4f\t%.4f\t%.4f'% (a,b,y[i]))
        print('-------------------------')
    return t, y

#In ein Array speichern für die Verallgemeinerung
#Absoluter Fehler noch plotten für den Vergleich

# SEP Aufgabe 2c
x = np.linspace(1, 6)
#plot
x_RungeKutta, y_RungeKutta = rungeKutta(f, 1, 6, 500, 5)
x_RungeKutta1, y_RungeKutta1 = rungeKuttaButcher(f, 1, 6, 500, 5)

plt.figure(1)
plt.plot(x, y(x), '-', x_RungeKutta, y_RungeKutta, x_RungeKutta1, y_RungeKutta1)
plt.legend(["exakt", "klass. RK","neues RK"])
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Runge-Kutta Verfahren')

plt.figure(2)
plt.semilogy(x_RungeKutta, np.abs(y(x_RungeKutta) - y_RungeKutta), x_RungeKutta1, np.abs(y(x_RungeKutta1) - y_RungeKutta1))
plt.legend(["Absoluter Fehler klass. RK","Absoluter Fehler neues RK"])
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Der erfundene RungeKutta Verfahren mit Butcher Tabelle liefert einen vernünftigen Wert
# Jedoch ist der Absolute Fehler grösser somit würde ich das normal verwenden

