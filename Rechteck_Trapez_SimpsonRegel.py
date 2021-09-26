# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:46:49 2021

@author: Selim Arslan
Rechteck-, Trapez- und Simpsonregel
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

"""
Exakte Integration

m = 10 kg
v = 20 m /s
a = 5 m/s

t = Integral 10 / (-v*(v**(1/2))) dv -> [20 / v**0.5] 
a = 5 und b = 20 bestimtes Integrall berechnen von [20 / v**0.5] 
4.472135954999580
"""
#%% Summierte Rechtecksformel
def sum_Rechteck(f, a, b, n):
    h = (b-a) / n
    res = 0.
    for i in range(n):
        res = res + f(a + (i + 0.5) * h)
    return h * res
        
        
def f(v):
    return 10 / (-v*(v**(1/2)))

a = 20
b = 5
n = 5
I = 0.544987

rf = sum_Rechteck(f, a, b, n)
absoluterFehler = np.absolute(I - rf)

print('Rf = ', rf)
print('Absulter Fehler = ',absoluterFehler)

#%% Summierte Trapezregel


def sum_Trapezregel(f, a, b, n):
    h = (b-a) / n
    res = 0.
    fTemp = (f(a) + f(b)) / 2
    for i in range(1,n):
        res = res + f(a + (i * h))
    return h * (fTemp + res)

def f(v):
    return 10 / (-v*(v**(1/2)))

a = 20
b = 5
n = 5
I = 0.544987

tf = sum_Rechteck(f, a, b, n)
absoluterFehler = np.absolute(I - tf)

print('Tf = ', tf)
print('Absulter Fehler = ',absoluterFehler)

#%% Summierte Simpsonregel

def sum_simpson(f, a, b, n):
    h = (b - a)/n
    res = 0.
    for i in range(n):
        res = res + f(a + (i + 0.5) * h)
    res = 2 * res
    for i in range(1,n):
        res = res + f(a +  i * h)
    res = res + 0.5 * (f(a) + f(b))
    return res *h/3.

def f(v):
    return 10 / (-v*(v**(1/2)))

a = 20
b = 5
n = 5
I = 0.544987

sf = sum_simpson(f, a, b, n)
absoluterFehler = np.absolute(I - sf)

print('Tf = ', sf)
print('Absulter Fehler = ',absoluterFehler)

#%% Summierte Trapezregel mit Daten und nicht äquidistante Werte
r = np.array([0, 800, 1200, 1400, 2000, 3000, 3400, 3600, 4000, 5000, 5500, 6370], dtype="float64")* 1000
roh = np.array([13000, 12900, 12700, 12000, 11650, 10600, 9900, 5500, 5300, 4750, 4500, 3300], dtype="float64")


#Summierte Trapezformel für nicht äquidistante Werte
def sum_Trapez(x, y):
    Tf_neq = 0
    if len(x) != len(y):
        raise AttributeError('Länge der Array stimmen nicht überrein')
    for i in range(len(x)-1):
        Tf_neq = Tf_neq + ((y[i] + y[i + 1]) / 2 * (x[i + 1] - x[i]))
    return Tf_neq

def f(r, roh):
    masse = roh * 4 * np.pi * r**2
    return masse

Tf_neq = sum_Trapez(r, f(r, roh))

#Konstante für die Erdmasse
Erdmasse = 5.9723e24

#Fehlerberechnung
absoluterFehler = np.absolute(Tf_neq - Erdmasse)
relativerFehler = np.absolute(absoluterFehler / Erdmasse * 100)

print('Erdmasse= ', Erdmasse)
print('Tf_neq= ', Tf_neq)
print('Absoluter Fehler= ', absoluterFehler)
print('Relativer Fehler= ', relativerFehler)
#%% 