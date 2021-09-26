# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:31:40 2021

@author: Selim Arslan
Interpolation mir allgemeine Lagrange-Verfahren
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Lagrange
def lagrange(x, y, x_int):
    y_int = 0
    
    for i in range(x.size):
        g = 1
        for j in range(y.size):
            if i != j:
                g = g * (x_int - x[j]) / (x[i] - x[j])
        y_int = y_int + g * y[i]
    return y_int

x = np.array([0, 2500, 5000, 10000])
y = np.array([1013, 747, 540, 226])
x_int = 3750

y_int = lagrange(x, y, x_int)
print('x-Wert = ', x_int)
print('y-Wert = ', y_int)

#%% Verwendung von Polyfit und PolyVal

x = np.array([1981, 1984, 1989, 1993, 1997, 2000, 2001, 2003, 2004, 2010])
y = np.array([0.5, 8.2, 15, 22.9, 36.6, 51, 56.3, 61.8, 65, 76.7])

z = np.polyfit(x, y, 9)#Koeffizienten des Interpolationspolynoms
p = np.poly1d(z)

x_Achse = np.arange(1975, 2020, 0.1)
plt.figure(1)
plt.plot(x_Achse, np.polyval(p, x_Achse))
plt.show()
#%% Schätzaufgabe
def lagrange(x, y, x_int):
    y_int = 0
    
    for i in range(x.size):
        g = 1
        for j in range(y.size):
            if i != j:
                g = g * (x_int - x[j]) / (x[i] - x[j])
        y_int = y_int + g * y[i]
    return y_int

x_int = 2020

y_int = lagrange(x, y, x_int)
print('x-Wert = ', x_int)
print('y-Wert = ', y_int)

"""
Ist nicht logisch und kann nicht benutzt werden, bei Interpolationen muss sich der gesuchte 
Wert im INtervall befinden. Das heisst unser bereich ist 1981<=gesucht<=2010 Alles darüber oder darunter ist
nicht realistisch mit unserer Lagrange Formel.
Lagrange-Formel sind ausserhalb des Intervall nicht verwendbar, da die Funktion keine Datenpunkte hatte die ausserhalb 
des INtervalls interpolieren konnte. Somit kann man nicht vorraussagen was über der Grenze passsiert.
"""
#%% 
x = np.array([1981, 1984, 1989, 1993, 1997, 2000, 2001, 2003, 2004, 2010])
y = np.array([0.5, 8.2, 15, 22.9, 36.6, 51, 56.3, 61.8, 65, 76.7])

z = np.polyfit(x, y, 9)
p = np.poly1d(z)


x_Neu = np.array([1981, 1984, 1989, 1993, 1997, 2000])
y_Neu = np.array([0.5, 8.2, 15, 22.9, 36.6, 51])

x_Achse = np.arange(1975, 2020, 0.1)
x_Achse_Neu = np.arange(1981, 2010, 0.1)

plt.figure(1)
plt.plot(x_Achse, np.polyval(p, x_Achse))
plt.plot(x_Achse_Neu, lagrange(x, y, x_Achse_Neu))
plt.show()
"""
Gleiche Begründung wie in Aufgabe 2, sobald sich die Werte nicht im Interfall befinden
werden die y-Werte realistisch
"""