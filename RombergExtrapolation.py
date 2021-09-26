# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:29:29 2021

@author: Selim Arslan
Romberg Extrapolation
"""

import numpy as np

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
    return 2 * np.exp(-1 * (x/10 - 2)**4)

x = romberg(f, 0, 40, 3)# Die 4 sagt wie lang die erste Spalte ist. je grösser desto genauer
print(x)