# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:12:02 2021

@author: Selim Arslan
Ausgleichsrechnung
"""

import numpy as np
import matplotlib.pyplot as plt
#%% Ausgleichsrechnung
T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
g = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4])

def f1(x):
    return x**2

def f2(x):
    return x

def f3(x):
    return np.ones(x.shape)

def f(x, lam):
    return lam[0] * f1(x) + lam[1] * f2(x) + lam[2] * f3(x)

A = np.zeros([11, 3])
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
plt.plot(x_fine, f(x_fine,lam))

plt.plot(x_fine, np.polyval(p, x_fine)) #Aufgabe C

#Plot labels
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.legend(['data', 'f(x) = a * x + b'])
plt.show()
#%% Ausgleichsrechnung mit Datensatz

data = np.array([[33.00, 53.00, 3.32, 3.42, 29.00],
        [31.00, 36.00, 3.10, 3.26, 24.00],
        [33.00, 51.00, 3.18, 3.18, 26.00],
        [37.00, 51.00, 3.39, 3.08, 22.00],
        [36.00, 54.00, 3.20, 3.41, 27.00],
        [35.00, 35.00, 3.03, 3.03, 21.00],
        [59.00, 56.00, 4.78, 4.57, 33.00],
        [60.00, 60.00, 4.72, 4.72, 34.00],
        [59.00, 60.00, 4.60, 4.41, 32.00],
        [60.00, 60.00, 4.53, 4.53, 34.00],
        [34.00, 35.00, 2.90, 2.95, 20.00],
        [60.00, 59.00, 4.40, 4.36, 36.00],
        [60.00, 62.00, 4.31, 4.42, 34.00],
        [60.00, 36.00, 4.27, 3.94, 23.00],
        [62.00, 38.00, 4.41, 3.49, 24.00],
        [62.00, 61.00, 4.39, 4.39, 32.00],
        [90.00, 64.00, 7.32, 6.70, 40.00],
        [90.00, 60.00, 7.32, 7.20, 46.00],
        [92.00, 92.00, 7.45, 7.45, 55.00],
        [91.00, 92.00, 7.27, 7.26, 52.00],
        [61.00, 62.00, 3.91, 4.08, 29.00],
        [59.00, 42.00, 3.75, 3.45, 22.00],
        [88.00, 65.00, 6.48, 5.80, 31.00],
        [91.00, 89.00, 6.70, 6.60, 45.00],
        [63.00, 62.00, 4.30, 4.30, 37.00],
        [60.00, 61.00, 4.02, 4.10, 37.00],
        [60.00, 62.00, 4.02, 3.89, 33.00],
        [59.00, 62.00, 3.98, 4.02, 27.00],
        [59.00, 62.00, 4.39, 4.53, 34.00],
        [37.00, 35.00, 2.75, 2.64, 19.00],
        [35.00, 35.00, 2.59, 2.59, 16.00],
        [37.00, 37.00, 2.73, 2.59, 22.00]])



A = np.array([[x[0], x[1], x[2], x[3], 1] for x in data])


x = np.arange(len(data))
y1 = data.T[4]

lambd = np.linalg.solve(A.T @ A, A.T @ y1)

y2 = np.array([lambd[0] * T_Tank + lambd[1] * T_Benzin + lambd[2] * p_Tank + lambd[3] * p_Benzin + lambd[4] for T_Tank, T_Benzin, p_Tank, p_Benzin, _ in data])

plt.scatter(x, y1, label="Messungen")
plt.scatter(x, y2, label="Approximation")

plt.bar(x, np.abs(y1-y2), label="Differenz")
plt.legend()
plt.show()
#%% 
# Moore0sches Gesetz
data=np.array([
[1971, 2250.],
[1972, 2500.],
[1974, 5000.],
[1978, 29000.],
[1982, 120000.],
[1985, 275000.],
[1989, 1180000.],
[1989, 1180000.],
[1993, 3100000.],
[1997, 7500000.],
[1999, 24000000.],
[2000, 42000000.],
[2002, 220000000.],
[2003, 410000000.],
])

plt.figure(1)
plt.semilogy(data[:,0],data[:,1],'+')
plt.grid(True), plt.xlabel('t / Jahr'), plt.ylabel('# Transistoren / Chip')
plt.title('Prozessorentwicklung / Moor''sches Gesetz')
plt.show()

##Aufgabe a)
##---------

n = data.shape[0]
t = data[:,0]
y = np.log10(data[:,1])

#Definition der Basisfunktion
def f1(t):
    return (1.)
def f2(t):
    return (t - 1970.)

#Definition der Ausgleichfunktion
def f(lambd, t):
    return 10.**(lambd[0] + (t-1970.)*lambd[1])

#Berechnung der Matrix A
A = np.zeros((n,2))
for i in range(0,n):
    A[i,:] = np.array([f1(t[i]), f2(t[i])])
    

[Q,R] = np.linalg.qr(A)
lambd = np.linalg.solve(R, Q.T @ y)

#Plot
t_new = np.arange(t[0], t[-1]+1)

plt.figure(2)
plt.semilogy(data[:,0], data[:,1], '+', t_new, f(lambd,t_new))
plt.grid(True), plt.xlabel('t / Jahr'), plt.ylabel('#Transisitor / Chrip')
plt.title('Prozessorentwicklung / Moore''sches Gesetz')
plt.legend(['data','fit'])
plt.show()

#Aufgabe B

fit2015 = f(lambd, 2015)

print('Vergleich 2015: ', fit2015, 4e9)

t_new = np.arange(t[0], 2020)
plt.figure(3)
plt.semilogy(data[:,0], data[:,1], '+', t_new, f(lambd, t_new), [2015], [4e9], 'bo')
plt.grid(True), plt.xlabel('t / Jahr'), plt.ylabel('#Transisitor / Chrip')
plt.title('Präprozessorentwicklung / Moore''Gesetz')
plt.legend(['data','fit', '2015'])
