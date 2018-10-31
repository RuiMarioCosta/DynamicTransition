#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:39:14 2017

@author: Rui Costa
"""

import numpy as np
import multiprocessing

from scipy.signal import argrelmax, argrelmin, sawtooth
from scipy.integrate import odeint, simps
from scipy.optimize import fmin, fsolve


import free_energy as free

muB = 5.7883818066 * (10**(-5))  # eV T^-1
kB = 8.6173324 * (10**(-5))  # eV K^-1


def c1(t, c10, a, b):
    return (c10 - b / (a + b)) * np.exp(-(a + b) * t) + b / (a + b)


def c_2(t, c10, k12, k21):

    def f1(y, t, k12, k21):
        c1, c2 = y
        return [-k12 * c1 + k21 * c2, k12 * c1 - k21 * c2]

    y0 = [c10, 1. - c10]
    return odeint(f1, y0, t, args=(k12, k21))


def c_3(t, c10, c20, k12, k21, k23, k32):

    def f2(y, t, k12, k21, k23, k32):
        c1, c2, c3 = y
        return [-k12 * c1 + k21 * c2, k12 * c1 -
                (k21 + k23) * c2 + k32 * c3, k23 * c2 - k32 * c3]

    y0 = [c10, c20, 1. - c10 - c20]
    return odeint(f2, y0, t, args=(k12, k21, k23, k32))


def c_4(t, c10, c20, c30, k12, k21, k23, k32, k34, k43):

    def f3(y, t, k12, k21, k23, k32, k34, k43):
        c1, c2, c3, c4 = y
        return [-k12 * c1 + k21 * c2,
                k12 * c1 - (k21 + k23) * c2 + k32 * c3,
                k23 * c2 - (k32 + k34) * c3 + k43 * c4,
                k34 * c3 - k43 * c4]

    y0 = [c10, c20, c30, 1. - c10 - c20 - c30]
    return odeint(f3, y0, t, args=(k12, k21, k23, k32, k34, k43))


def T_applied(t, dt, T_min, T_max, Ti, dT):
    T = np.zeros_like(t) + Ti
    n = 1  # variable used for step increase
    for i in xrange(len(t)):
        if t[i] > n * dt:
            n += 1
            T[i:] += dT
            if T[i] < T_min or T[i] > T_max:
                dT = -dT
    return T


def B_applied(t, dt, B0, Bi, dB):
    """Applied magnetic field.

    Parameters
    ----------
    t : array
        Time
    dt : scalar
        Time step between jumps
    B0 : scalar
        Maximum and minimum (limits) of the applied magnetic field
    Bi : scalar
        Initial magnetic field
    dB : scalar
        Magnetic field step between jumps

    Returns
    -------
    y : array
        Triangular function.
    """
    B = np.zeros_like(t) + Bi
    n = 1  # variable used for step increase
    for i in xrange(len(t)):
        if t[i] > n * dt:
            n += 1
            B[i:] += dB
            if B0 <= np.abs(B[i]):
                dB = -dB
    return B


def time_redimension(v, tom):
    """Calculates order of magnitude of time based on the probabilities so
    that kij = vij * pij is approximatelly 1, and that way overflow or zero
    probabilities do not occur.

    Parameters
    ----------
    v : array, list
        Rate frequency per second
    tom : scalar
        Order of magnitude of time.

    Returns
    -------
    y : array
        Exponents in base e. The redimensioned frequency, v', is v*10^(tom).
    """
    a = v * 0.  # array to store values of exponents
    for i in xrange(len(v)):
        a[i] = np.log(v[i]) + tom * np.log(10.)

    return a


def concentrations_magnetization(
        t,
        tom,
        T_min,
        T_max,
        Ti,
        dt_T,
        dT,
        B0,
        Bi,
        dt_B,
        dB,
        sigma,
        J1,
        J2,
        gJ,
        Tc1,
        Tc2,
        Nm,
        ThetaD1,
        ThetaD2,
        N,
        F01,
        F02):
    T = T_applied(t, dt_T, T_min, T_max, Ti, dT)
    B = B_applied(t, dt_B, B0, Bi, dB)
    concent = np.zeros((len(t), 6))  # array to store concentration values
    magnetization = np.zeros_like(t)  # array to store magentization values
    print 'T =', T[0], 'B =', B[0]

    # initial concentrations
    free_1 = free.free_total_vs_M(
        sigma, T[0], B[0], J1, gJ, Tc1, Nm, ThetaD1, N, F01)
    free_2 = free.free_total_vs_M(
        sigma, T[0], B[0], J2, gJ, Tc2, Nm, ThetaD2, N, F02)
    free_tot_stable = free.free_total_stable_vs_M(
        sigma, T[0], B[0], J1, J2, gJ, Tc1, Tc2, Nm, ThetaD1, ThetaD2, N,
        F01, F02, free_1, free_2)

    min1_indices = argrelmin(free_1)[0]
    min2_indices = argrelmin(free_2)[0]
    min_indices = argrelmin(free_tot_stable)[0]  # indices of minimums
#    print '1', min1_indices, '2', min2_indices, 'all', min_indices
    if Tc2 <= T[0]:
        if min(free_1) <= min(free_2):
            concent[0] = [1., 0., 0., 0., 0., 0.]  # PM1 state
        else:
            concent[0] = [0., 1., 0., 0., 0., 0.]  # PM2 state
    elif Tc1 <= T[0] < Tc2:
        if free_1[min1_indices] < min(free_2[min2_indices]):
            concent[0] = [1., 0., 0., 0., 0., 0.]  # PM1 state
        else:
            if B[0] < 0.:
                concent[0] = [0., 0., 0., 0., 1., 0.]  # negative FM2 state
            else:
                concent[0] = [0., 0., 0., 0., 0., 1.]  # positive FM2 state
    else:
        if min(free_1[min1_indices]) < min(free_2[min2_indices]):
            if B[0] < 0.:
                concent[0] = [0., 0., 1., 0., 0., 0.]  # negative FM1 state
            else:
                concent[0] = [0., 0., 0., 1., 0., 0.]  # positive FM1 state
        else:
            if B[0] < 0.:
                concent[0] = [0., 0., 0., 0., 1., 0.]  # negative FM2 state
            else:
                concent[0] = [0., 0., 0., 0., 0., 1.]  # positive FM2 state

    magnetization[0] = sigma[np.argmin(free_tot_stable)]

    # calculation of the evolution of the concentrations
    for i in xrange(1, len(t) - 1):
        Ti = T[i]
        Bi = B[i]
        print '---------------', i, t[i], Ti, Bi, '--------'
        free_1 = free.free_total_vs_M(
            sigma, Ti, Bi, J1, gJ, Tc1, Nm, ThetaD1, N, F01)
        free_2 = free.free_total_vs_M(
            sigma, Ti, Bi, J2, gJ, Tc2, Nm, ThetaD2, N, F02)
        free_tot_stable = free.free_total_stable_vs_M(
            sigma, Ti, Bi, J1, J2, gJ, Tc1, Tc2, Nm, ThetaD1, ThetaD2, N,
            F01, F02, free_1, free_2)

        min_indices = argrelmin(free_tot_stable)[0]  # indices of minimums
        max_indices = argrelmax(free_tot_stable)[0]  # indices of maximums
        min1_indices = argrelmin(free_1)[0]  # indices of minimums
        min2_indices = argrelmin(free_2)[0]  # indices of minimums

        beta = 1. / (kB * Ti)
        Z = simps(np.exp(-beta * free_tot_stable), sigma)
        sigma2i = simps((sigma**2.)*np.exp(-beta * free_tot_stable), sigma) / Z
        sigma_avgi = simps(sigma*np.exp(-beta * free_tot_stable), sigma) / Z
        sigma2[i] = sigma2i
        sigma_avg[i] = sigma_avgi
        print sigma2i, sigma_avgi
#        print 'indices :', min_indices, max_indices, min1_indices, min2_indices
        print 'Z=', Z, 'beta=', beta
#        print

        v = 1e9  # atttempt rate per second
        v12, v21, v23, v32, v34, v43 = v, v, v, v, v, v
        k12, k21, k23, k32, k34, k43 = None, None, None, None, None, None
        
        if Tc2 <= Ti:  # if Ti if bigger than both Curie temperatures
            if len(min_indices) == 1:
                print '====== 1 minimum : Tc2 <= Ti'
                if min(free_2) < min(free_1):  # if it is the PM2 state
                    concent[i] = [0., 1., 0., 0., 0., 0.]
                else:  # if it is the PM1 state
                    concent[i] = [1., 0., 0., 0., 0., 0.]

                magnetization[i] = sigma[min_indices]

            if len(min_indices) == 2:  # both PM states
                print '====== 2 minimums : Tc2 <= Ti'
                min_free_1, min_free_2 = free_tot_stable[min_indices]
                max_free_12 = free_tot_stable[max_indices][0]

                Delta_free_12 = max_free_12 - min_free_1  # energy difference
                Delta_free_21 = max_free_12 - min_free_2

                a12 = beta * Delta_free_12  # probability exponent
                a21 = beta * Delta_free_21
                
                v = np.array([v12, v21])  # attempt frequencies (in seconds)
                a = np.array([a12, a21])  # exponent of probabilities

                a_rate = time_redimension(v, tom)

#                print 'v', v, 'a', a, 'tom', tom
#                print 'a_r', a_rate
#                print 'a_r - a', a_rate - a
#                print 'exp', np.exp(a_rate - a)

                k12, k21 = np.exp(a_rate - a) / Z

                if k12 == float("inf") or k21 == float("inf"):
                    a = a_rate - a
                    c2 = 1. / (1. + np.exp(a[1] - a[0]))
                    c1 = 1. - c2
                    if sigma[min1_indices] < sigma[min2_indices]:
                        concent[i] = [c1, c2, 0., 0., 0., 0.]
                    else:
                        concent[i] = [c2, c1, 0., 0., 0., 0.]
                else:
                    if sigma[min1_indices] < sigma[min2_indices]:
                        k12 = k12 * (1. - np.exp(-concent[i-1,0]))
                        k21 = k21 * (1. - np.exp(-concent[i-1,1]))
                        sol = c_2(t[i:i + 2], concent[i - 1, 0], k12, k21)
                        concent[i] = [sol[1, 0], sol[1, 1], 0., 0., 0., 0.]
                    else:
                        k12 = k12 * (1. - np.exp(-concent[i-1,1]))
                        k21 = k21 * (1. - np.exp(-concent[i-1,0]))
                        sol = c_2(t[i:i + 2], concent[i - 1, 1], k12, k21)
                        concent[i] = [sol[1, 1], sol[1, 0], 0., 0., 0., 0.]

                magnetization[i] = sigma[min1_indices] * \
                    concent[i, 0] + sigma[min2_indices] * concent[i, 1]

        elif Tc1 <= Ti < Tc2:
            if len(min_indices) == 3:
                print '====== 3 minimums : Tc1 <= Ti < Tc2'

                min_free_1, min_free_2, min_free_3 = free_tot_stable[min_indices]
                max_free_12, max_free_23 = free_tot_stable[max_indices]

                Delta_free_12 = max_free_12 - min_free_1
                Delta_free_21 = max_free_12 - min_free_2
                Delta_free_23 = max_free_23 - min_free_2
                Delta_free_32 = max_free_23 - min_free_3

                a12 = beta * Delta_free_12  # probability exponent
                a21 = beta * Delta_free_21
                a23 = beta * Delta_free_23
                a32 = beta * Delta_free_32

                # attempt frequencies (in seconds)
                v = np.array([v12, v21, v23, v32])
                a = np.array([a12, a21, a23, a32])  # exponent of probabilities

                a_rate = time_redimension(v, tom)

                # transition probabilities per second
                k12, k21, k23, k32 = np.exp(a_rate - a) / Z
#                k12 = k12 / Delta_free_12
#                k21 = k21 / Delta_free_21
#                k23 = k23 / Delta_free_23
#                k32 = k32 / Delta_free_32

                if k12 == float("inf") or k21 == float(
                        "inf") or k23 == float("inf") or k32 == float("inf"):
                    a = a_rate - a
                    c2 = 1. / (1. + np.exp(a[1] - a[0]) + np.exp(a[2] - a[3]))
                    c1 = c2 * np.exp(a[1] - a[0])
                    c3 = 1. - c1 - c2
                    concent[i] = [c2, 0., 0., 0., c1, c3]
                    magnetization[i] = sigma[min1_indices] * concent[i, 0] + \
                        sigma[min2_indices[0]] * concent[i, 4] + \
                        sigma[min2_indices[1]] * concent[i, 5]
                else:
                    sol = c_3(t[i:i + 2], concent[i - 1, 4],
                              concent[i - 1, 0], k12, k21, k23, k32)
                    concent[i] = [sol[1, 1], 0., 0., 0., sol[1, 0], sol[1, 2]]
                    magnetization[i] = sigma[min1_indices] * concent[i, 0] + \
                        sigma[min2_indices[0]] * concent[i, 4] + \
                        sigma[min2_indices[1]] * concent[i, 5]

            elif len(min_indices) == 2:
                print '====== 2 minimums : Tc1 <= Ti < Tc2'
                min_free_1, min_free_2 = free_tot_stable[min_indices]
                max_free_12 = free_tot_stable[max_indices][0]

                Delta_free_12 = max_free_12 - min_free_1
                Delta_free_21 = max_free_12 - min_free_2

                a12 = beta * Delta_free_12  # probability exponent
                a21 = beta * Delta_free_21

                v = np.array([v12, v21])  # attempt frequencies (in seconds)
                a = np.array([a12, a21])  # exponent of probabilities

                a_rate = time_redimension(v, tom)

                # transition probabilities per second
                k12, k21 = np.exp(a_rate - a) / Z
#                k12 = k12 / Delta_free_12
#                k21 = k21 / Delta_free_21
                
                if k12 == float("inf") or k21 == float("inf"):
                    a = a_rate - a
                    c2 = 1. / (1. + np.exp(a[1] - a[0]))
                    c1 = 1. - c2
                    if len(
                            min2_indices) == 2 and min_indices[0] == min2_indices[0] and min_indices[1] == min2_indices[1]:
                        concent[i] = [0., 0., 0., 0., c1, c2]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 4] + sigma[min_indices[1]] * concent[i, 5]
                    elif Bi < 0.:  # if PM1 and negative FM2 states
                        concent[i] = [c2, 0., 0., 0., c1, 0.]
                        magnetization[i] = sigma[min_indices[1]] * \
                            concent[i, 0] + sigma[min_indices[0]] * concent[i, 4]
                    else:  # if PM1 and positive FM2 states
                        concent[i] = [c1, 0., 0., 0., 0., c2]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 0] + sigma[min_indices[1]] * concent[i, 5]
                else:
                    # if both minimums are from system 2
                    if len(
                            min2_indices) == 2 and min_indices[0] == min2_indices[0] and min_indices[1] == min2_indices[1]:
                        sol = c_2(t[i:i + 2], concent[i - 1, 4], k12, k21)
                        concent[i] = [0., 0., 0., 0., sol[1, 0], sol[1, 1]]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 4] + sigma[min_indices[1]] * concent[i, 5]
                    elif Bi < 0.:  # if PM1 and negative FM2 states
                        sol = c_2(t[i:i + 2], concent[i - 1, 4], k12, k21)
                        concent[i] = [sol[1, 1], 0., 0., 0., sol[1, 0], 0.]
                        magnetization[i] = sigma[min_indices[1]] * \
                            concent[i, 0] + sigma[min_indices[0]] * concent[i, 4]
                    else:  # if PM1 and positive FM2 states
                        sol = c_2(t[i:i + 2], concent[i - 1, 0], k12, k21)
                        concent[i] = [sol[1, 0], 0., 0., 0., 0., sol[1, 1]]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 0] + sigma[min_indices[1]] * concent[i, 5]

            else:  # if there is only 1 minimum
                print '====== 1 minimum : Tc1 <= Ti < Tc2'
                if min_indices == min1_indices:
                    concent[i] = [1., 0., 0., 0., 0., 0.]
                else:
                    if Bi < 0.:
                        concent[i] = [0., 0., 0., 0., 0., 1.]
                    else:
                        concent[i] = [0., 0., 0., 0., 1., 0.]
                magnetization[i] = sigma[min_indices]
        else:
            if len(min_indices) == 1:
                print '====== 1 minimum : Ti < Tc1'
                if Bi < 0.:
                    if min_indices == min1_indices:
                        concent[i] = [0., 0., 1., 0., 0., 0.]
                    else:
                        concent[i] = [0., 0., 0., 0., 1., 0.]
                else:
                    if min_indices == min1_indices:
                        concent[i] = [0., 0., 0., 1., 0., 0.]
                    else:
                        concent[i] = [0., 0., 0., 0., 0., 1.]
                magnetization[i] = sigma[min_indices]

            elif len(min_indices) == 2:
                print '====== 2 minimums : Ti < Tc1'
                min_free_1, min_free_2 = free_tot_stable[min_indices]
                max_free_12 = free_tot_stable[max_indices][0]

                Delta_free_12 = max_free_12 - min_free_1  # Energy difference
                Delta_free_21 = max_free_12 - min_free_2

                a12 = beta * Delta_free_12  # probability exponent
                a21 = beta * Delta_free_21

                v = np.array([v12, v21])  # attempt frequencies (in seconds)
                a = np.array([a12, a21])  # exponent of probabilities

                a_rate = time_redimension(v, tom)

#                print 'v:', v, 'a:', a, 'tom:', tom
#                print 'a_r:', a_rate
#                print 'a_r - a:', (a_rate - a)[0], (a_rate - a)[1]
#                print 'exp:', np.exp(a_rate - a)

                k12, k21 = np.exp(a_rate - a) / Z

                if k12 == float("inf") or k21 == float("inf"):
                    a = a_rate - a
                    c2 = 1. / (1. + np.exp(a[1] - a[0]))
                    c1 = 1. - c2
                    if min_indices[0] == min2_indices[0] and min_indices[1] == min2_indices[1]:
                        concent[i] = [0., 0., 0., 0., c1, c2]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 4] + sigma[min_indices[1]] * concent[i, 5]
                    elif min_indices[0] == min1_indices[0] and min_indices[1] == min1_indices[1]:
                        concent[i] = [0., 0., c1, c2, 0., 0.]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 2] + sigma[min_indices[1]] * concent[i, 3]
                    elif min_indices[0] == min2_indices[0] and min_indices[1] == min1_indices[0]:
                        concent[i] = [0., 0., c2, 0., c1, 0.]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 4] + sigma[min_indices[1]] * concent[i, 2]
                    elif min_indices[0] == min2_indices[0] and min_indices[1] == min1_indices[1]:
                        concent[i] = [0., 0., 0., c2, c1, 0.]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 4] + sigma[min_indices[1]] * concent[i, 3]
                    elif min_indices[0] == min1_indices[0] and min_indices[1] == min2_indices[1]:
                        concent[i] = [0., 0., c1, 0., 0., c2]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 2] + sigma[min_indices[1]] * concent[i, 5]
                    else:
                        concent[i] = [0., 0., 0., c1, 0., c2]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 3] + sigma[min_indices[1]] * concent[i, 5]
                else:
                    if min_indices[0] == min2_indices[0] and min_indices[1] == min2_indices[1]:
                        sol = c_2(t[i:i + 2], concent[i - 1, 4], k12, k21)
                        concent[i] = [0., 0., 0., 0., sol[1, 0], sol[1, 1]]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 4] + sigma[min_indices[1]] * concent[i, 5]
                    elif min_indices[0] == min1_indices[0] and min_indices[1] == min1_indices[1]:
                        sol = c_2(t[i:i + 2], concent[i - 1, 2], k12, k21)
                        concent[i] = [0., 0., sol[1, 0], sol[1, 1], 0., 0.]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 2] + sigma[min_indices[1]] * concent[i, 3]
                    elif min_indices[0] == min2_indices[0] and min_indices[1] == min1_indices[0]:
                        sol = c_2(t[i:i + 2], concent[i - 1, 4], k12, k21)
                        concent[i] = [0., 0., sol[1, 1], 0., sol[1, 0], 0.]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 4] + sigma[min_indices[1]] * concent[i, 2]
                    elif min_indices[0] == min2_indices[0] and min_indices[1] == min1_indices[1]:
                        sol = c_2(t[i:i + 2], concent[i - 1, 4], k12, k21)
                        concent[i] = [0., 0., 0., sol[1, 1], sol[1, 0], 0.]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 4] + sigma[min_indices[1]] * concent[i, 3]
                    elif min_indices[0] == min1_indices[0] and min_indices[1] == min2_indices[1]:
                        sol = c_2(t[i:i + 2], concent[i - 1, 2], k12, k21)
                        concent[i] = [0., 0., sol[1, 0], 0., 0., sol[1, 1]]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 2] + sigma[min_indices[1]] * concent[i, 5]
                    else:
                        sol = c_2(t[i:i + 2], concent[i - 1, 3], k12, k21)
                        concent[i] = [0., 0., 0., sol[1, 0], 0., sol[1, 1]]
                        magnetization[i] = sigma[min_indices[0]] * \
                            concent[i, 3] + sigma[min_indices[1]] * concent[i, 5]

            elif len(min_indices) == 3:
                print '====== 3 minimums : Ti < Tc1'

                min_free_1, min_free_2, min_free_3 = free_tot_stable[min_indices]
                max_free_12, max_free_23 = free_tot_stable[max_indices]

                Delta_free_12 = max_free_12 - min_free_1
                Delta_free_21 = max_free_12 - min_free_2
                Delta_free_23 = max_free_23 - min_free_2
                Delta_free_32 = max_free_23 - min_free_3

                a12 = beta * Delta_free_12  # probability exponent
                a21 = beta * Delta_free_21
                a23 = beta * Delta_free_23
                a32 = beta * Delta_free_32

                # attempt frequencies (in seconds)
                v = np.array([v12, v21, v23, v32])
                a = np.array([a12, a21, a23, a32])  # exponent of probabilities

                a_rate = time_redimension(v, tom)

                # transition probabilities per second
                k12, k21, k23, k32 = np.exp(a_rate - a) / Z

                if k12 == float("inf") or k21 == float(
                        "inf") or k23 == float("inf") or k32 == float("inf"):
                    a = a_rate - a
                    c2 = 1. / (1. + np.exp(a[1] - a[0]) + np.exp(a[2] - a[3]))
                    c1 = c2 * np.exp(a[1] - a[0])
                    c3 = 1. - c1 - c2
                    if Bi < 0.:
                        concent[i] = [0., 0., c2, 0., c1, c3]
                        magnetization[i] = sigma[min_indices[0]] * concent[i, 4] + \
                            sigma[min_indices[1]] * concent[i, 2] + \
                            sigma[min_indices[2]] * concent[i, 5]
                    else:
                        concent[i] = [0., 0., 0., c2, c1, c3]
                        magnetization[i] = sigma[min_indices[0]] * concent[i, 4] + \
                            sigma[min_indices[1]] * concent[i, 3] + \
                            sigma[min_indices[2]] * concent[i, 5]
                else:
                    if Bi < 0.:
                        sol = c_3(t[i:i + 2], concent[i - 1, 4],
                                  concent[i - 1, 2], k12, k21, k23, k32)
                        concent[i] = [0., 0., sol[1, 1],
                                      0., sol[1, 0], sol[1, 2]]
                        magnetization[i] = sigma[min_indices[0]] * concent[i, 4] + \
                            sigma[min_indices[1]] * concent[i, 2] + \
                            sigma[min_indices[2]] * concent[i, 5]
                    else:
                        sol = c_3(t[i:i + 2], concent[i - 1, 4],
                                  concent[i - 1, 3], k12, k21, k23, k32)
                        concent[i] = [0., 0., 0, sol[1, 1],
                                      sol[1, 0], sol[1, 2]]
                        magnetization[i] = sigma[min_indices[0]] * concent[i, 4] + \
                            sigma[min_indices[1]] * concent[i, 3] + \
                            sigma[min_indices[2]] * concent[i, 5]

            else:
                print '====== 4 minimums : Ti < Tc1'

                min_free_1, min_free_2, min_free_3, min_free_4 = free_tot_stable[min_indices]
                max_free_12, max_free_23, max_free_34 = free_tot_stable[max_indices]

                Delta_free_12 = max_free_12 - min_free_1  # Energy difference
                Delta_free_21 = max_free_12 - min_free_2
                Delta_free_23 = max_free_23 - min_free_2
                Delta_free_32 = max_free_23 - min_free_3
                Delta_free_34 = max_free_34 - min_free_3
                Delta_free_43 = max_free_34 - min_free_4

                a12 = beta * Delta_free_12  # probability exponent
                a21 = beta * Delta_free_21
                a23 = beta * Delta_free_23
                a32 = beta * Delta_free_32
                a34 = beta * Delta_free_34
                a43 = beta * Delta_free_43

                # attempt frequencies (in seconds)
                v = np.array([v12, v21, v23, v32, v34, v43])
                # exponent of probabilities
                a = np.array([a12, a21, a23, a32, a34, a43])

                a_rate = time_redimension(v, tom)

                k12, k21, k23, k32, k34, k43 = np.exp(
                    a_rate - a) / Z  # transition probabilities per second

                if k12 == float("inf") or k21 == float("inf") or k23 == float(
                        "inf") or k32 == float("inf") or k34 == float("inf") or k43 == float("inf"):
                    a = a_rate - a
                    c2 = 1. / \
                        (1. + np.exp(a[1] - a[0]) + np.exp(a[2] - a[3]) + np.exp(a[2] - a[3] + a[4] - a[5]))
                    c1 = c2 * np.exp(a[1] - a[0])
                    c3 = c2 * np.exp(a[2] - a[3])
                    c4 = 1. - c1 - c2 - c3
                    concent[i] = [0., 0., c2, c3, c1, c4]
                    magnetization[i] = sigma[min_indices[0]] * concent[i, 4] + \
                        sigma[min_indices[1]] * concent[i, 2] + \
                        sigma[min_indices[2]] * concent[i, 3] + \
                        sigma[min_indices[3]] * concent[i, 5]

                else:
                    sol = c_4(t[i:i + 2], concent[i - 1, 4], concent[i - 1, 2],
                              concent[i - 1, 3], k12, k21, k23, k32, k34, k43)
                    concent[i] = [0., 0., sol[1, 1],
                                  sol[1, 2], sol[1, 0], sol[1, 3]]
                    magnetization[i] = sigma[min_indices[0]] * concent[i, 4] + \
                        sigma[min_indices[1]] * concent[i, 2] + \
                        sigma[min_indices[2]] * concent[i, 3] + \
                        sigma[min_indices[3]] * concent[i, 5]

        print
        print "k's :", k12, k21, k23, k32, k34, k43

    return concent, magnetization


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from variables import *

    dt = 0.01
    t = np.arange(0, 10 + dt, dt)
    tom = 25.  # time order of magnitude

    T_min = 40.
    T_max = 320.
    Ti = 300.
    dt_T = 200.
    dT = 1.

    Bi = 0.
    B0 = 1.
    dt_B = 0.025
    dB = 0.01

    dsigma = 1e-3
    sigma = np.arange(-1, 1 + dsigma, dsigma)

    plot_concent = 1
    plot_mags = 1
    
    sigma2 = np.zeros_like(t)
    sigma_avg = np.zeros_like(t)


#    plt.figure('c_2')
#    plt.plot(t, c1(t, 1., 1., 1.), 'o', label='analitic')
#    plt.plot(t, c_2(t, 0., 1., 1.), label='numeric')
#    plt.legend(loc=0, fontsize='xx-small')
#    plt.ylim(0, 1)
#    plt.xlim(xmin=0)
#
#    plt.figure('c_3')
#    plt.plot(t, c_3(t, 0.2, 0.3, 5., 1., 5., 3.), label='numeric')
#    plt.legend(loc=0, fontsize='xx-small')
#    plt.ylim(0, 1)
#    plt.xlim(xmin=0)
#
#    plt.figure('c_4')
#    plt.plot(t, c_4(t, 0.2, 0.3, 0.25, 5., 1., 5., 3., 1., 1.), label='numeric')
#    plt.legend(loc=0, fontsize='xx-small')
#    plt.ylim(0, 1)
#    plt.xlim(xmin=0)

    fig = plt.figure('plots')

    plt.subplot(231)
#    plt.figure('T_applied')
    plt.plot(t, T_applied(t, dt_T, T_min, T_max, Ti, dT))
    plt.legend(loc=0, fontsize='xx-small')
    plt.xlim(xmin=0)
    plt.xlabel('t(s $\cdot$ $10^{%d}$)' % tom)
    plt.ylabel('T(K)')
    plt.grid()

    plt.subplot(232)
#    plt.figure('B_applied')
    plt.plot(t, B_applied(t, dt_B, B0, Bi, dB))
    plt.legend(loc=0, fontsize='xx-small')
    plt.xlim(xmin=0)
    plt.xlabel('t(s $\cdot$ $10^{%d}$)' % tom)
    plt.ylabel('B(T)')
    plt.grid()

    if plot_concent:
        concent, magnetization = concentrations_magnetization(t,
                                                              tom,
                                                              T_min,
                                                              T_max,
                                                              Ti,
                                                              dt_T,
                                                              dT,
                                                              B0,
                                                              Bi,
                                                              dt_B,
                                                              dB,
                                                              sigma,
                                                              J,
                                                              J,
                                                              gJ,
                                                              Tc1,
                                                              Tc2,
                                                              Nm,
                                                              ThetaD1,
                                                              ThetaD2,
                                                              N,
                                                              DeltaF,
                                                              0.)
#        plt.figure('concentrations')

        plt.subplot(233)
#        plt.plot(t, concent[:, 0], 's', ms=16., label='PM1')
#        plt.plot(t, concent[:, 1], 'o', ms=14., label='PM2')
#        plt.plot(t, concent[:, 2], 's', ms=10., label='FM1_1')
#        plt.plot(t, concent[:, 3], 'o', ms=8., label='FM1_2')
#        plt.plot(t, concent[:, 4], 's', ms=4., label='FM2_1')
#        plt.plot(t, concent[:, 5], 'o', ms=2., label='FM2_2')

        plt.plot(t, concent[:, 0], '.', ms=1., label='PM1')
        plt.plot(t, concent[:, 1], '.', ms=1., label='PM2')
        plt.plot(t, concent[:, 2], '.', ms=1., label='FM1_1')
        plt.plot(t, concent[:, 3], '.', ms=1., label='FM1_2')
        plt.plot(t, concent[:, 4], '.', ms=1., label='FM2_1')
        plt.plot(t, concent[:, 5], '.', ms=1., label='FM2_2')

        plt.legend(loc=0, fontsize='xx-small')
        plt.xlim(xmin=0)
        plt.grid()
        plt.xlabel('t(s $\cdot$ $10^{%d}$)' % tom)
        plt.ylabel('concentrations')

        if plot_mags:
            plt.subplot(234)
#            plt.figure('magnetization')
            plt.plot(t, magnetization)
            plt.legend(loc=0, fontsize='xx-small')
            plt.xlim(xmin=0)
            plt.grid()
            plt.xlabel('t(s $\cdot$ $10^{%d}$)' % tom)
            plt.ylabel('$\sigma$')

            plt.subplot(235)
#            plt.figure('M vs B')
            plt.plot(B_applied(t, dt_B, B0, Bi, dB), magnetization)
            plt.legend(loc=0, fontsize='xx-small')
            plt.grid()
            plt.xlabel('B(T)')
            plt.ylabel('$\sigma$')

            plt.subplot(236)
#            plt.figure('M vs T')
            plt.plot(T_applied(t, dt_T, T_min, T_max, Ti, dT), magnetization)
            plt.legend(loc=0, fontsize='xx-small')
            plt.grid()
            plt.xlabel('T(K)')
            plt.ylabel('$\sigma$')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.tight_layout(pad=0., w_pad=-3, h_pad=-1, rect=[-0.03, 0, 1, 1])
    
    plt.figure('sigmas')
    plt.subplot(121)
    plt.plot(t, sigma2)
    plt.grid()
    plt.ylabel('$<\sigma^2>$')
    plt.xlabel('t(s $\cdot$ $10^{%d}$)' % tom)
    
    plt.subplot(122)
    plt.plot(t, sigma_avg)
    plt.grid()
    plt.ylabel('$\sigma$')
    plt.xlabel('t(s $\cdot$ $10^{%d}$)' % tom)
    
    plt.show()
    
    print sigma2
