# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 08:08:42 2021

@author: Selim Arslan
"""
import numpy as np
import matplotlib.pyplot as plt

def sum_Trapezregel(f, a, b, n):
    h = (b-a) / n
    res = 0.
    fTemp = (f(a) + f(b)) / 2
    for i in range(1,n):
        res = res + f(a + (i * h))
    return h * (fTemp + res)


def f(x):
    return np.sin(x)

h = 0.056
a = 0
b = np.pi
n = int((b-a)/ h)
I = 2;

tf = sum_Trapezregel(f, a, b, n)
print('Tf = ', tf)

absoluterFehler = np.absolute(I - tf)
print('Absulter Fehler = ',absoluterFehler)