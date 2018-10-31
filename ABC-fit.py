# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:08:49 2017

@author: Rui
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def poly5(x, a0, a1, a2, a3, a4, a5):
    return a0 + a1*x + a2*x**2. + a3*x**3. + a4*x**4. + a5*x**5.

def poly4(x, a0, a1, a2, a3, a4):
    return a0 + a1*x + a2*x**2. + a3*x**3. + a4*x**4.

def poly3(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*x**2. + a3*x**3.

def poly2(x, a0, a1, a2):
    return a0 + a1*x + a2*x**2.

def poly1(x, a0, a1):
    return a0 + a1*x

T = [192.5,193.5,194.5,195.5,196.5]
A = [208.60799, 255.67453, 283.11433, 321.01766, 354.68402]
B = [-0.04884, -0.05194, -0.04432, -0.04512, -0.04527]
C = [2.92479e-6, 2.99348e-6, 2.34125e-6, 2.29662e-6, 2.2277e-6]

dt = 0.1
T_fit = np.arange(192.5, 196.5+dt, dt)

# poly4
popt, pcov = curve_fit(poly4, T, A)
print '\n--poly4--\n', popt
A_fit4 = poly4(T_fit, *popt)
popt, pcov = curve_fit(poly4, T, B)
print popt
B_fit4 = poly4(T_fit, *popt)
popt, pcov = curve_fit(poly4, T, C)
print popt
C_fit4 = poly4(T_fit, *popt)

# poly3
popt, pcov = curve_fit(poly3, T, A)
print '\n--poly3--\n', popt
A_fit3 = poly3(T_fit, *popt)
popt, pcov = curve_fit(poly3, T, B)
print popt
B_fit3 = poly3(T_fit, *popt)
popt, pcov = curve_fit(poly3, T, C)
print popt
C_fit3 = poly3(T_fit, *popt)

# poly2
popt, pcov = curve_fit(poly2, T, A)
print '\n--poly2--\n', popt
A_fit2 = poly2(T_fit, *popt)
popt, pcov = curve_fit(poly2, T, B)
print popt
B_fit2 = poly2(T_fit, *popt)
popt, pcov = curve_fit(poly2, T, C)
print popt
C_fit2 = poly2(T_fit, *popt)

# poly1
popt, pcov = curve_fit(poly1, T, A)
print '\n--poly1--\n', popt
A_fit1 = poly1(T_fit, *popt)
popt, pcov = curve_fit(poly1, T, B)
print popt
B_fit1 = poly1(T_fit, *popt)
popt, pcov = curve_fit(poly1, T, C)
print popt
C_fit1 = poly1(T_fit, *popt)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(T, C, '--o', label='data')
ax1.plot(T_fit, C_fit1, label='poly1')
ax1.plot(T_fit, C_fit2, label='poly2')
ax1.plot(T_fit, C_fit3, label='poly3')
ax1.plot(T_fit, C_fit4, label='poly4')
ax1.set_ylabel('C(Oe g^5/emu^5)')
ax1.grid()
ax1.legend(loc=0)

ax2.plot(T, B, '--o', label='data')
ax2.plot(T_fit, B_fit1, label='poly1')
ax2.plot(T_fit, B_fit2, label='poly2')
ax2.plot(T_fit, B_fit3, label='poly3')
ax2.plot(T_fit, B_fit4, label='poly4')
ax2.set_ylabel('B(Oe g^3/emu^3)')
ax2.grid()
ax2.legend(loc=0)

ax3.plot(T, A, '--o', label='data')
ax3.plot(T_fit, A_fit1, label='poly1')
ax3.plot(T_fit, A_fit2, label='poly2')
ax3.plot(T_fit, A_fit3, label='poly3')
ax3.plot(T_fit, A_fit4, label='poly4')
ax3.set_ylabel('A(Oe g/emu)')
ax3.set_xlabel('T(K)')
ax3.grid()
ax3.legend(loc=0)

plt.show()