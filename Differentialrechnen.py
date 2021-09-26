# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:42:46 2021

@author: Selim Arslan
Differentialgleichung
"""

import numpy as np
import matplotlib.pyplot as plt
 
#%% Energie Plotten
g = 9.81
s0 = 0
v0 = 100
m = 1
 
def v(t): return -g*t+v0
def s(t): return -0.5*g*t**2 + v0*t + s0
 
t = np.linspace(0,20)
 
plt.figure(1)
plt.plot(t,v(t))
plt.plot(t,s(t))
plt.grid(), plt.legend(['s(t)', 'v(t)']), plt.xlabel('Time t [s]')
plt.show()
 
def E_kin(t): return 0.5*m*v(t)**2
def E_pot(t): return m * g * s(t)
def E_tot(t): return E_kin(t) + E_pot(t)
 
plt.figure(2)
plt.plot(t,E_kin(t), t, E_pot(t), t, E_tot(t))
plt.grid(), plt.legend(['E_kin', 'E_pot', 'E_tot'])
plt.xlabel('Time t [s]'), plt.ylabel('Energy [J]')
plt.show()

#%% Romberg wie lang zum Bremsen in sekunden

def romberg(f, a, b, n):


    r = np.array( [[0] * (n+1)] * (n+1), float )
    h = b - a
    r[0,0] = 0.5 * h * ( f( a ) + f( b ) )

    powerOf2 = 1
    for i in range(1, n + 1):

       
        h = 0.5 * h

        sum = 0.0
        powerOf2 = 2 * powerOf2
        for k in range( 1, powerOf2, 2 ):
            sum = sum + f( a + k * h )


        r[i,0] = 0.5 * r[i-1,0] + sum * h

        powerOf4 = 1
        for j in range(1, i + 1):
            powerOf4 = 4 * powerOf4
            r[i,j] = r[i,j-1] + (r[i,j-1] - r[i-1,j-1]) / (powerOf4 - 1)

    return r

#Funktion
def f(x):
    return 97000 / (-5*x**2 - 570000)


x = romberg(f, 100, 0, 4)# Die 4 sagt wie lang die erste Spalte ist. je grösser desto genauer
print(x)

#%% Romberg Bremsweg

def romberg(f, a, b, n):


    r = np.array( [[0] * (n+1)] * (n+1), float )
    h = b - a
    r[0,0] = 0.5 * h * ( f( a ) + f( b ) )

    powerOf2 = 1
    for i in range(1, n + 1):

       
        h = 0.5 * h

        sum = 0.0
        powerOf2 = 2 * powerOf2
        for k in range( 1, powerOf2, 2 ):
            sum = sum + f( a + k * h )


        r[i,0] = 0.5 * r[i-1,0] + sum * h

        powerOf4 = 1
        for j in range(1, i + 1):
            powerOf4 = 4 * powerOf4
            r[i,j] = r[i,j-1] + (r[i,j-1] - r[i-1,j-1]) / (powerOf4 - 1)

    return r

#Funktion
def f(x):
    return (97000 / (-5*x**2 - 570000)) * x


x = romberg(f, 100, 0, 4)# Die 4 sagt wie lang die erste Spalte ist. je grösser desto genauer
print(x)

#%% Raketengleichung
g = 9.81
vRel = 2600
mA = 300000
mE = 80000
tE = 190

u = (mA - mE) / tE

def sum_Trapezregel(f, a, b, n):
    h = (b-a) / n
    res = 0.
    fTemp = (f(a) + f(b)) / 2
    for i in range(1,n):
        res = res + f(a + (i * h))
    return h * (fTemp + res)


def a(t):
    return vRel * (u / (mA - u * t)) - g

def v(t):
    return sum_Trapezregel(a, 0., t, 5)

def h(t):
    return sum_Trapezregel(v, 0., t, 5)


t = np.linspace(0., tE + 1)

Beschleunigung = a(tE)
Geschwindigkeit = v(tE)
höhe = h(tE)
print('Beschleunigung= ', Beschleunigung)
print('Geschwindigkeit= ', Geschwindigkeit)
print('Höhe= ', höhe)

plt.figure(1)
plt.plot(t, a(t))
plt.grid()
plt.title('Beschleunigung a(t)')
plt.xlabel('Zeit t [s]')
plt.ylabel('Beschleunigung [m/s^2]')

plt.figure(2)
plt.plot(t, v(t))
plt.grid()
plt.title('Geschwindigkeit v(t)')
plt.xlabel('Zeit t [s]')
plt.ylabel('Geschwindigkeit in m/s')

plt.figure(3)
plt.plot(t, h(t))
plt.grid()
plt.title('Höhe h(t)')
plt.xlabel('Zeit t [s]')
plt.ylabel('Höhe [m]')
plt.show()
#%% Analytische Lösung

g = 9.81
vRel = 2600
mA = 300000
mE = 80000
tE = 190

u = (mA - mE) / tE

def v(t):
    return vRel * np.log(mA / (mA - u * t)) - g * t

def h(t):
    return - ((vRel * (mA - u * t)) / u) * np.log(mA / (mA - u * t)) + vRel * t - 0.5 * g * t**2 


t = np.linspace(0,tE)

plt.figure(4)
plt.plot(t, v(t), t, h(t))
plt.grid()
plt.title('Geschwindigkeit v(t) und Höhe h(t) mit analytischer Lösung')
plt.legend(['v(t)', 'h(t)'])
plt.xlabel('Time t [s]')
plt.ylabel('Geschwindigkeit [m/s]')
 
plt.show()

