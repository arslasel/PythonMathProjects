# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:43:17 2020

Höhere Mathematik 1, Serie 11, Aufgabe 3 (Gerüst)

@author: Gruppe4_Piologin_Arslasel
"""
import numpy as np
import matplotlib.pyplot as plt

"""
    Normalform z = x + iy
    
    C = x + iy
    
    Z0 = 0
    Zn+1 = Zn^2 + C
"""

detail = 1000                       #number of pixels in x and y direction
maxit = 100                          #maximum n for iterations
x_min = -2                         #minimim value of x-interval
x_max = 0.7                        #maximum value of x-interval
y_min = -1.4                       #minimim vale of y-interval
y_max = 1.4                      #minimim vale of y-interval


a = np.linspace(x_min, x_max , detail, dtype=np.float64)  #define real axis [x_min,x_max]
b = np.linspace(y_min, y_max, detail, dtype=np.float64)  #define imaginary axis [y_min,y_max]

B = np.zeros((detail,detail))        #for color valzues n 

[x,y] = np.meshgrid(a, b)      #to create the complex plane with the axes defined by a and b


C = x+y*1j                           #creating the plane
Z = np.zeros(np.shape(C), np.complex64)  #initial conditions (first iteration), Z has same dimension as C
for n in np.arange(1,maxit+1):       #start iteration
  Z = Z * Z + C                      #calculating Z
  expl = np.where(abs(Z) > 2)         #finding exploded values (i.e. with an absolute value > 2)
  Z[expl] = 0                      #removing from iteration
  C[expl] = 0                        #removing from plane
  B[expl] = n                        #saving color value n

plt.figure(1)
B = B/np.max(np.max(B))           #deviding by max value for correct color
plt.imshow(B,extent=[x_min,x_max,y_min,y_max],origin='lower',interpolation='bilinear')   #display image

"""
Plot zoom figure 1
"""
detail = 1000                       #number of pixels in x and y direction
maxit = 100                          #maximum n for iterations
x_min = -0.5                        #minimim value of x-interval
x_max = 0.5                        #maximum value of x-interval
y_min = -0.2                      #minimim vale of y-interval
y_max = 0.2                    #minimim vale of y-interval


a = np.linspace(x_min, x_max , detail, dtype=np.float64)  #define real axis [x_min,x_max]
b = np.linspace(y_min, y_max, detail, dtype=np.float64)  #define imaginary axis [y_min,y_max]

B = np.zeros((detail,detail))        #for color valzues n 

[x,y] = np.meshgrid(a, b)      #to create the complex plane with the axes defined by a and b


C = x+y*1j                           #creating the plane
Z = np.zeros(np.shape(C), np.complex64)  #initial conditions (first iteration), Z has same dimension as C
for n in np.arange(1,maxit+1):       #start iteration
  Z = Z * Z + C                      #calculating Z
  expl = np.where(abs(Z) > 2)         #finding exploded values (i.e. with an absolute value > 2)
  Z[expl] = 0                      #removing from iteration
  C[expl] = 0                        #removing from plane
  B[expl] = n                        #saving color value n

plt.figure(2)
B = B/np.max(np.max(B))           #deviding by max value for correct color
plt.imshow(B,extent=[x_min,x_max,y_min,y_max],origin='lower',interpolation='bilinear')   #display image


"""
Plot zoom figure 1 
"""
detail = 1000                       #number of pixels in x and y direction
maxit = 100                          #maximum n for iterations
x_min = 0.2                         #minimim value of x-interval
x_max = 0.4                        #maximum value of x-interval
y_min = -0.1                       #minimim vale of y-interval
y_max = 0.1                      #minimim vale of y-interval


a = np.linspace(x_min, x_max , detail, dtype=np.float64)  #define real axis [x_min,x_max]
b = np.linspace(y_min, y_max, detail, dtype=np.float64)  #define imaginary axis [y_min,y_max]

B = np.zeros((detail,detail))        #for color valzues n 

[x,y] = np.meshgrid(a, b)      #to create the complex plane with the axes defined by a and b


C = x+y*1j                           #creating the plane
Z = np.zeros(np.shape(C), np.complex64)  #initial conditions (first iteration), Z has same dimension as C
for n in np.arange(1,maxit+1):       #start iteration
  Z = Z * Z + C                      #calculating Z
  expl = np.where(abs(Z) > 2)         #finding exploded values (i.e. with an absolute value > 2)
  Z[expl] = 0                      #removing from iteration
  C[expl] = 0                        #removing from plane
  B[expl] = n                        #saving color value n

plt.figure(3)
B = B/np.max(np.max(B))           #deviding by max value for correct color
plt.imshow(B,extent=[x_min,x_max,y_min,y_max],origin='lower',interpolation='bilinear')   #display image

"""
Plot zoom figure1
"""

"""
Plot zoom figure 1 
"""
detail = 1000                       #number of pixels in x and y direction
maxit = 100                          #maximum n for iterations
x_min = 0.25                       #minimim value of x-interval
x_max = 0.30                     #maximum value of x-interval
y_min = -0.015                      #minimim vale of y-interval
y_max = 0.015                      #minimim vale of y-interval


a = np.linspace(x_min, x_max , detail, dtype=np.float64)  #define real axis [x_min,x_max]
b = np.linspace(y_min, y_max, detail, dtype=np.float64)  #define imaginary axis [y_min,y_max]

B = np.zeros((detail,detail))        #for color valzues n 

[x,y] = np.meshgrid(a, b)      #to create the complex plane with the axes defined by a and b


C = x+y*1j                           #creating the plane
Z = np.zeros(np.shape(C), np.complex64)  #initial conditions (first iteration), Z has same dimension as C
for n in np.arange(1,maxit+1):       #start iteration
  Z = Z * Z + C                      #calculating Z
  expl = np.where(abs(Z) > 2)         #finding exploded values (i.e. with an absolute value > 2)
  Z[expl] = 0                      #removing from iteration
  C[expl] = 0                        #removing from plane
  B[expl] = n                        #saving color value n

plt.figure(4)
B = B/np.max(np.max(B))           #deviding by max value for correct color
plt.imshow(B,extent=[x_min,x_max,y_min,y_max],origin='lower',interpolation='bilinear')   #display image



"""
Plot zoom figure 1 
"""
detail = 1000                       #number of pixels in x and y direction
maxit = 100                          #maximum n for iterations
x_min = 0.28                       #minimim value of x-interval
x_max = 0.29                     #maximum value of x-interval
y_min = -0.010                     #minimim vale of y-interval
y_max = -0.015                      #minimim vale of y-interval


a = np.linspace(x_min, x_max , detail, dtype=np.float64)  #define real axis [x_min,x_max]
b = np.linspace(y_min, y_max, detail, dtype=np.float64)  #define imaginary axis [y_min,y_max]

B = np.zeros((detail,detail))        #for color valzues n 

[x,y] = np.meshgrid(a, b)      #to create the complex plane with the axes defined by a and b


C = x+y*1j                           #creating the plane
Z = np.zeros(np.shape(C), np.complex64)  #initial conditions (first iteration), Z has same dimension as C
for n in np.arange(1,maxit+1):       #start iteration
  Z = Z * Z + C                      #calculating Z
  expl = np.where(abs(Z) > 2)         #finding exploded values (i.e. with an absolute value > 2)
  Z[expl] = 0                      #removing from iteration
  C[expl] = 0                        #removing from plane
  B[expl] = n                        #saving color value n

plt.figure(5)
B = B/np.max(np.max(B))           #deviding by max value for correct color
plt.imshow(B,extent=[x_min,x_max,y_min,y_max],origin='lower',interpolation='bilinear')   #display image


"""
Plot zoom figure 1 
"""
detail = 1000                       #number of pixels in x and y direction
maxit = 100                          #maximum n for iterations
x_min = 0.285                       #minimim value of x-interval
x_max = 0.287                     #maximum value of x-interval
y_min = -0.012                     #minimim vale of y-interval
y_max = -0.011                      #minimim vale of y-interval


a = np.linspace(x_min, x_max , detail, dtype=np.float64)  #define real axis [x_min,x_max]
b = np.linspace(y_min, y_max, detail, dtype=np.float64)  #define imaginary axis [y_min,y_max]

B = np.zeros((detail,detail))        #for color valzues n 

[x,y] = np.meshgrid(a, b)      #to create the complex plane with the axes defined by a and b


C = x+y*1j                           #creating the plane
Z = np.zeros(np.shape(C), np.complex64)  #initial conditions (first iteration), Z has same dimension as C
for n in np.arange(1,maxit+1):       #start iteration
  Z = Z * Z + C                      #calculating Z
  expl = np.where(abs(Z) > 2)         #finding exploded values (i.e. with an absolute value > 2)
  Z[expl] = 0                      #removing from iteration
  C[expl] = 0                        #removing from plane
  B[expl] = n                        #saving color value n

plt.figure(6)
B = B/np.max(np.max(B))           #deviding by max value for correct color
plt.imshow(B,extent=[x_min,x_max,y_min,y_max],origin='lower',interpolation='bilinear')   #display image
