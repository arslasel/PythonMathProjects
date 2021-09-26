# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:53:30 2021

@author: Selim Arslan
RungeKutta-Verfahren
"""

import numpy as np
import matplotlib.pyplot as plt


#%%RungeKutta mit Funktion

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
        #Y-Wert Berechnen
        y[i + 1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        print('%.4f\t%.4f\t%.4f'% (a,b,y[i]))
        print('-------------------------')
    return x,y


x = np.linspace(1, 6)
#plot
plt.figure(1)
x_RungeKutta, y_RungeKutta = rungeKutta(f, 1, 6, 83, 5)
plt.plot(x, y(x), '-', x_RungeKutta, y_RungeKutta, '--,')
plt.legend(['Runge-Kutta'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid('major')
plt.title('Runge-Kutta Verfahren')

#%% RungeKutta Verfahren mit neue Butchertabelle

def f(t,y):
    return 1 - (y/t)

def y(t):
    return (t / 2) + 9 / (2*t)

def rungeKuttaButcher(f, a, b, n, y0):
    h = (b-a) / n
    x = np.linspace(a, b, n+1)
    y = np.empty(n + 1)
    y[0] = y0
    
    for i in range(n):
        #Steigung berechnen
        #Steigungen berechnen
        k1 = f(x[i], y[i])
        k2 = f(x[i] + 0.25 * h, y[i] + 0.5 * h * k1)
        k3 = f(x[i] + 0.5 * h, y[i] + 0.75 * h * k2)
        k4 = f(x[i] + 0.75 * h, y[i] + h * k3)
        #Y-Wert Berechnen
        y[i + 1] = y[i] + h * (k1 + 2*k2 + 3*k3 + 4*k4) / 10
        
        #Numerischen Wert anzeigen
        print('%.4f\t%.4f\t%.4f'% (a,b,y[i]))
        print('-------------------------')
    return x, y

#In ein Array speichern für die Verallgemeinerung
#Absoluter Fehler noch plotten für den Vergleich

x = np.linspace(1, 6)
#plot
x_RungeKutta, y_RungeKutta = rungeKutta(f, 1, 6, 500, 5)
x_RungeKutta1, y_RungeKutta1 = rungeKuttaButcher(f, 1, 6, 500, 5)
plt.plot(x, y(x), '-', x_RungeKutta, y_RungeKutta, x_RungeKutta1, y_RungeKutta1)
plt.legend(['Runge-Kutta', 'Normal', 'Eigenwerte'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid('major')
plt.title('Runge-Kutta Verfahren')
plt.show()

err_10 = np.abs(y_RungeKutta - y(x_RungeKutta)) 
err_20 = np.abs(y_RungeKutta1 - y(x_RungeKutta1))

print('err_10 = ', err_10[-1])
print('err_20 = ', err_20[-1])

plt.figure()
plt.plot(x_RungeKutta, err_10, x_RungeKutta1, err_20)
plt.grid()
plt.xlabel('err')
plt.ylabel('y')
plt.grid('major')
plt.title('Absoluter Fehler')
plt.legend(['Runga-Kutta', 'Butcher'])
plt.show()

#%%
def f(x, y): 
    return x ** 2 / y

def sol(t):
    return np.sqrt((2 * t ** 3) / 3 + 4)

def eulerVerfahren(f, a, b, n, y0):
    h = (b - a) / n
    y = [y0 for _ in range(n + 1)]
    for i in range(n):
        xi = a + i * h
        y[i + 1] = y[i] + h * f(xi, y[i])
    return y


def mittelwert_eulerVerfahren(f, a, b, n, y0):
    h = (b - a) / n
    y = [y0 for _ in range(n + 1)]
    for i in range(n):
        xi = a + i * h
        xh2 = xi + 0.5 * h
        yh2 = y[i] + 0.5 * h * f(xi, y[i])
        y[i + 1] = y[i] + h * f(xh2, yh2)
    return y


def modifiziertes_eulerVerfahren(f, a, b, n, y0):
    h = (b - a) / n
    y = [y0 for _ in range(n + 1)]

    for i in range(n):
        xi = a + i * h
        k1 = f(xi, y[i])
        yE = y[i] + h * k1
        k2 = f(xi + h, yE)
        y[i + 1] = y[i] + h * 0.5 * (k1 + k2)
    return y


def runge_kuttaVerfahren(f, a, b, n, y0):
    h = (b - a) / n
    y = [y0 for _ in range(n + 1)]

    for i in range(n):
        xi = a + i * h
        k1 = f(xi, y[i])
        k2 = f(xi + h / 2, y[i] + h / 2 * k1)
        k3 = f(xi + h / 2, y[i] + h / 2 * k2)
        k4 = f(xi + h, y[i] + h * k3)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def arslasel_S12_Aufg3(f, a, b, n, y0):
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    return x, eulerVerfahren(f, a, b, n, y0), mittelwert_eulerVerfahren(f, a, b, n, y0), modifiziertes_eulerVerfahren(f, a, b, n, y0), runge_kuttaVerfahren(f,a,b,n, y0)
                                                                                                                    
                                                                                                                    
                                                                                                                   
a, b, n, y0 = 0, 10, 100, 2
x, y_euler, y_mittelpunkt, y_modeuler, y_rk = arslasel_S12_Aufg3(f, a, b, n, y0)


solu = [np.sqrt((2 * t ** 3) / 3 + 4) for t in x]

x_small = np.linspace(a, b, 100)
plt.figure(1)
plt.plot(x, solu, label="Exakt")
plt.plot(x, y_euler, ':', label="Euler")
plt.plot(x, y_mittelpunkt, 'g-.', label="Mittelpunkt")
plt.plot(x, y_modeuler, 'r--', label="Mod. Euler")
plt.plot(x, y_rk, 'm--', label="Runge-Kutta")
plt.legend()
plt.xlabel('Iterationen')
plt.grid()
plt.title('Aufgabe 3: Funktionenvergleich')
plt.show()

err_euler = [abs(solu[i] - y_euler[i]) for i in range(len(x))]
err_mittelp = [abs(solu[i] - y_mittelpunkt[i]) for i in range(len(x))]
err_modeuler = [abs(solu[i] - y_modeuler[i]) for i in range(len(x))]
err_rk = [abs(solu[i] - y_rk[i]) for i in range(len(x))]
plt.figure(2)
plt.plot(x, err_euler)
plt.semilogy(x, err_euler, label="Euler")
plt.semilogy(x, err_mittelp, label="Mittelpunkt")
plt.semilogy(x, err_modeuler, label="Mod. Euler")
plt.semilogy(x, err_rk, label="Runge-Kutta")
plt.title('Aufgabe 3: Globalen Fehler')
plt.legend()
plt.grid()
plt.xlabel('Iterationen')
plt.ylabel('Fehler')
plt.show()