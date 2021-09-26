# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:52:38 2021

@author: Selim Arslan
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Ausgleichsrechnung
T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
g = np.array([76, 92, 106, 123, 137, 151, 179, 203, 227, 250, 281, 309])

def f1(x):
    return x**3

def f2(x):
    return x**2

def f3(x):
    return x

def f4(x):
    return np.ones(x.shape)

def f(x, lam):
    return lam[0] * f1(x) + lam[1] * f2(x) + lam[2] * f3(x) + lam[3] * f4(x)

A = np.zeros([12, 4])
A[:,0] = f1(T)
A[:,1] = f2(T)
A[:,2] = f3(T)
A[:,3] = f4(T)


def con(A):
    return np.linalg.norm(A,2) * np.linalg.norm(np.linalg.inv(A), 2)

def fehlerFunktional(f, yy, xx, lamda):
    result = 0
    for i in range(0, len(yy)):
        result += (yy[i] - f(lamda, xx[i])) ** 2
    return result

[Q,R] = np.linalg.qr(A)
lam = np.linalg.solve(R, Q.T @ g)
print(lam)



x_fine = np.arange(0, 100, 0.1)


#Aufgabe C
z = np.polyfit(T, g, 10)
p = np.poly1d(z)




#plot Figure
plt.figure(1)
plt.plot(T, g)
plt.plot(x_fine, f(x_fine,lam))

plt.plot(T, g, 'o', x_fine, np.polyval(p, x_fine)) #Aufgabe C

#Plot labels
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.legend(['data', 'f(x) = a * x + b'])
plt.show()

#%%
T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
g = np.array([76, 92, 106, 123, 137, 151, 179, 203, 227, 250, 281, 309])

def f1(x):
    return x**2

def f2(x):
    return x

def f3(x):
    return np.ones(x.shape)

def f(x, lam):
    return lam[0] * f1(x) + lam[1] * f2(x) + lam[2] * f3(x)

A = np.zeros([12, 3])
A[:,0] = f1(T)
A[:,1] = f2(T)
A[:,2] = f3(T)



def con(A):
    return np.linalg.norm(A,2) * np.linalg.norm(np.linalg.inv(A), 2)

def fehlerFunktional(f, yy, xx, lamda):
    result = 0
    for i in range(0, len(yy)):
        result += (yy[i] - f(lamda, xx[i])) ** 2
    return result

[Q,R] = np.linalg.qr(A)
lam = np.linalg.solve(R, Q.T @ g)
print(lam)



x_fine = np.arange(0, 100, 0.1)


#Aufgabe C
z = np.polyfit(T, g, 10)
p = np.poly1d(z)




#plot Figure
plt.figure(1)
plt.plot(T, g)
plt.plot(T, g, 'o',x_fine, f(x_fine,lam))

plt.plot(x_fine, np.polyval(p, x_fine)) #Aufgabe C

#Plot labels
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.legend(['data', 'f(x) = a * x + b'])
plt.show()
