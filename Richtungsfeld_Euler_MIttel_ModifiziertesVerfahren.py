# -*- coding: utf-8 -*-
"""
Created on Sat May 29 09:39:11 2021

@author: Selim Arslan
Richtungsfeld
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Richtungsfeld
def f(x,y):
    return x**2 + 0.1 * y


def richtungsfeld(f, xMin, xMax, yMin, yMax, hx, hy):
   #Erzeuge Koordinaten auf der x, y Ebenen.
   x, y = np.meshgrid(np.linspace(xMin, xMax, int((xMax-xMin)/hx)) + 1, np.linspace(yMin, yMax, int((yMax-yMin)/hy) + 1))
   #Normberechnen
   n = np.sqrt(1 + f(x,y)**2) #Norm berechnen (Miesch Fragen)
   u = 1/n #Einheitsvektor
   v = f(x,y)/n #länge vom Vektor 
   #Steigungsdreieck berechen
   #Plot
   plt.quiver(x, y, u, v, color='blue', width=0.04)
   plt.show()
   return x,y

h = 0.3    
richtungsfeld(f, -1.5, 1.5, -1.5, 1.5, h, h)

#%% Euler, Mittelwert und Modifiziertes-Verfahren
def f(t,y):
    return t**2 + 0.1 * y

def sol(t):
    return -10*(t**2) - (200 * t) - 2000 + 1722.5 * np.exp(0.05 * (2*t + 3))

def euler(f, x0, y0, xn, n):
    h = (xn - x0)/ n
    x = np.linspace(x0, xn, n + 1)
    y = np.empty(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i],y[i])
        
    x_euler = np.linspace(x0, xn)
    #plot
    plt.plot(x_euler, sol(x_euler), x, y) # Euler-Verfahren wird ausgegeben
    plt.legend(['Exakter Wert', 'modifiziertesVerfahren'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid('major')
    plt.title('Euler-Verfahren')
    plt.show()
    return x,y


def mittelWert(f, x0, y0, n, a, b):
    x = x0
    y = y0
    h = (b-a)/n
    x_list = [x]
    y_list = [y]
    for i in range(n):
        yd = f(x,y)
        xm = x + h * 0.5
        ym = y + h * 0.5 * yd
        ydm = f(xm, ym)
        x = x + h
        y = y + h * ydm
        print('x(' + str(i+1) + '):', x)
        print('y(' + str(i+1) + '):', y)
        x_list.append(x)
        y_list.append(y)
        x_range = np.arange(a,b+0.1,0.1)
        
    plt.plot(x_range, sol(x_range))
    plt.plot(x_list,y_list)
    plt.legend(['Exakter Wert','Mittelpunktverfahren'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid('major')
    plt.title('Mittelw-Verfahren')
    plt.show()
    return x,y, x_list, y_list
    
    
def modifiziertes(f, a, b, y0, n):
    h = (b-a)/n
    t = np.zeros(n+1)
    y = np.zeros(n+1)
    
    t[0] = a
    y[0] = y0
    
    for i in range(y0, n):
        k1 = f(t[i], y[i])
        y_euler = y[i] + h * k1
        t[i+1] = t[i] + h
        k2 = f(t[i+1], y_euler)
        y[i+1] = y[i] + h * (k1 + k2) / 2
    
    t_new = np.arange(a, b,h/100)
    #Plot
    plt.plot(t,y,t_new,sol(t_new))
    plt.legend(['modifiziertesVerfahren','Exakter Wert'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid('major')
    plt.title('Modifiziertes Eulerverfahren')
    plt.show()
    return t_new, t, y
    
def richtungsfeld(f, xmin, xmax, ymin, ymax, hx, hy, x_euler, y_euler, x_list_mittel, y_list_mittel, x_mod, y_mod):
    x,y = np.meshgrid(np.linspace(xmin,xmax, int((xmax-xmin)/hx)+1), np.linspace(ymin,ymax, int((ymax-ymin)/hy)+1))
    n = np.sqrt(1+f(x,y)**2) 
    u = 1/n
    v = f(x,y)/n
    plt.quiver(x,y,u,v, color = 'blue', width=0.01)
    x_ = np.linspace(xmin,xmax)
    plt.plot(x_,sol(x_), color='red')
    plt.plot(x_euler, f(x_euler,y_euler), color="green")
    plt.plot(x_list_mittel,y_list_mittel, color="black")
    plt.plot(x_mod, f(x_mod,y_mod), color="darkred")
    plt.legend(['Exakter-Wert', 'Eulerverfahren', 'Mittelwertverfahren', 'modifiziertes Eulerver'])
    plt.title('Richtungsfeld mit allen Verfahren')
    plt.show()
    return x,y
    

def arslasel_S11_Aufg3(f, a, b, n, x0, y0):
    #Eulerverfahren
    x_euler, y_euler = euler(f, a, y0, b, n)
    #Mittelwertverfahren
    x_Mittel, y_Mittel, x_Array, y_Array = mittelWert(f, x0, y0, n, a, b)
    #Modifiziertes Eulerverfahren
    interval, x_modi, y_modi = modifiziertes(f, a, b, y0, n)
    
    h = 0.5
    
    
    #plot
    richtungsfeld(f, -1.5, 1.5, -1.5, 1.5, h, h, x_euler, y_euler, x_Mittel, y_Mittel, x_modi, y_modi)
    
    
arslasel_S11_Aufg3(f, -1.5, 1.5, 5, -1.5, 0)