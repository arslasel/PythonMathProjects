# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:33:38 2021

@author: Selim Arslan
Serie1 Aufgabe 1
"""

"""
Funktionen:
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plotFigure(v0, alpha, W):
    
    """
    3-Dimensionaler Plot mit WireFrame()
    """
    fig = plt.figure(0)
    Axes3D = fig.add_subplot(111, projection='3d')
    Axes3D.plot_wireframe(v0,alpha,W, rstride=5, cstride=5)

    plt.title('Gitter')
    Axes3D.set_xlabel('v0')
    Axes3D.set_ylabel('alpha')
    Axes3D.set_zlabel('Wurfweite')
    
    plt.show()
    
    
    """
    3-Dimesionaler Plot mit surface() und passender Colormap
    """
    fig = plt.figure(1)
    Axes3D = fig.add_subplot(111, projection='3d')
    surf = Axes3D.plot_surface(v0,alpha,W, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.title('Gitter')
    Axes3D.set_xlabel('v0')
    Axes3D.set_ylabel('alpha')
    Axes3D.set_zlabel('Wurfweite')
    
    plt.show()
    
    """
    2-Dimensionaler Plot mit Höhenlinie
    """
    fig = plt.figure(2)
    cont = plt.contour(v0, alpha, W, cmap=cm.coolwarm)
    fig.colorbar(cont, shrink=0.5, aspect=5)

    plt.title('Höhenlinien')
    plt.xlabel('v0')
    plt.ylabel('aplha')

    plt.show()
    
    
    
#%% Aufgabe a
g = 9.81
[v0,alpha] = np.meshgrid(np.linspace(0,100,100), np.linspace(0, np.pi / 2, 100)); #Gitter in die Ebene v0 und Alpha legen 
W = (v0**2 * np.sin(2*alpha)) / g #Berechne jeden Wert im Gitter.
plotFigure(v0, alpha, W)

#%% Aufgabe b1
R = 8.31
[V,T] = np.meshgrid(np.linspace(0.02, 0.2, 99), np.linspace(0, 10**4, 100));
p = (R*T) / V
plotFigure(V, T, p)
#%% Aufgabe b2
[p,T] = np.meshgrid(np.linspace(10**4, 10**5, 100), np.linspace(0.01, 10**4, 100));
V = (R*T) / p
plotFigure(p, T, V)
#%% Aufgabe b3
[p,V] = np.meshgrid(np.linspace(10**4, 10**6, 100), np.linspace(0, 10, 100));
T = (p*V) / R
plotFigure(p, V, T)

#%% Aufgabe 2
c = 1

def w(x, t):
    return np.sin(x + c*t)

[x,t] = np.meshgrid(np.linspace(0,15), np.linspace(0,15));
w = w(x,t)

fig = plt.figure(3)
ax = fig.add_subplot(111, projection = '3d')
ax.plot_wireframe(x,t,w, rstride=5, cstride=5)
#Beschriftung der Koordinaten
plt.title('Ausbreitung der Wellen')
ax.set_xlabel('X-Achse')
ax.set_ylabel('Zeit in Sekunden')
ax.set_zlabel('Auslenkung in Meter')
#Plot anzeigen
plt.show()

#%% Aufgabe 2.2
def v(x,t):
    return np.sin(x + c*t) + np.cos(2*x + 2*c*t)
[x,t] = np.meshgrid(np.linspace(0,15), np.linspace(0,15))
v = v(x,t)
fig = plt.figure(4)
ax = fig.add_subplot(111, projection = '3d')
ax.plot_wireframe(x,t,w, rstride=5, cstride=5)
#Beschriftung der Koordinaten
plt.title('Ausbreitung der Wellen')
ax.set_xlabel('X-Achse')
ax.set_ylabel('Zeit in Sekunden')
ax.set_zlabel('Auslenkung in Meter')
#Plot anzeigen
plt.show()

