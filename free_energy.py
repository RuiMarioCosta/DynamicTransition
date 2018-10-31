#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:16:53 2017

@author: Rui Costa
"""

import numpy as np
from scipy.integrate import quad
from scipy.signal import argrelmin, argrelmax

import magnetization as mag


muB = 5.7883818066 * (10**(-5))  # eV T^-1
kB = 8.6173324 * (10**(-5))  # eV K^-1


# Magnetic Free Energy
def free_M(T, B, J, gJ, Tc, Nm):
    """Magnetic free energy of 1 spin.

    Parameters
    ----------
    T : scalar
        Temperatures.
    B : scalar
        Magnetic fields.
    J : scalar
        Total angular momentum.
    gJ : scalar
        Landé g-factor.
    Tc : scalar
        Curie temperature.
    Nm : scalar
        Number of spins.

    Returns
    -------
    y : scalar
        Magnetic free energy.
    """

    Ms = Nm * gJ * muB * J  # Saturation magnetization
    lamb = 3 * kB * Tc / (gJ * muB * (J + 1) * Ms)

    # relative field to the saturation magnetization
    h = B / (lamb * Ms)
    sigma = mag.fm_magnetization(T, B, J, gJ, Tc, Nm)  # reduced magnetization
    y = 3. * J / (J + 1.) * (h + sigma) * Tc / T
    A = np.sinh((2. * J + 1.) * y / (2. * J))
    B = np.sinh(y / (2. * J))

    return -T * kB * np.log(A / B)  # magnetic free energy of 1 spin


# Lattice Free Energy
def free_L(T, thetaD):
    """Free Lattice Energy according to the Debye Model.

    Parameters
    ----------
    T : scalar, 1D array
        Temperature.
    theta_D : scalar
        Debye temperature of the material

    Returns
    --------
    y : scalar, array
        Free Lattice Energy
    """

    # Function in the integral
    def f(x):
        return (x**3.) / (np.exp(x) - 1.)

    integral = np.zeros_like(T)  # variable that stores the values

    if integral.shape == ():  # if T is just a single Temperature
        integral = quad(f, 0., thetaD / T)[0]
    else:  # if T is an array of Temperatures
        for i, t in enumerate(
                T):  # calculate the integral for each temperature
            integral[i] = quad(f, 0., thetaD / t)[0]

    return kB * (9. / 8. * thetaD - 3. * T * ((T / thetaD)**3.) *
                 integral + 3. * T * np.log(1. - np.exp(-thetaD / T)))


# Free Energy (faster calculation)
def free_total(T, B, J, gJ, Tc, Nm, ThetaD, N, F0):
    """Total free energy as a functions of temperature and magnetic field in the
    unit cell.

    Parameters
    ---------
    T : scalar
        Temperatures.
    B : scalar
        Magnetic fields.
    J : scalar
        Total angular momentum.
    ThetaD : scalar
        Debye temperature.
    F0 : scalar
        Electronic free enery at 0 K.

    Returns
    -------
    y : 2D array
        Total free energy.
    """
    f_M = free_M(T, B, J, gJ, Tc, Nm)  # magnetic free energy of 1 spin

    f_L = free_L(T, ThetaD) * np.ones_like(T)  # lattice free energy of 1 atom

    # h = B/(lamb*Nm*gJ*mu_B*J) # relative field to the saturation
    # magnetization
    F_0 = Nm * (3. * J / (J + 1.) * (0. + 1.) * kB * Tc) - N * kB * 9. / \
        8. * ThetaD + F0  # offset of free energies to make F start at F0

    return Nm * f_M + N * f_L + F_0


# Magnetic Free Energy as a function of Magnetization
def free_M_vs_M(sigma, T, B, J, gJ, Tc, Nm):
    """Magnetic Free Energy as a functio of Reduced Magnetization.

    Parameters
    ----------
    sigma : scalar, 1D array
        Reduced Magnetization.
    T : scalar
        Temperature.
    B : scalar
        Applied Magnetic Field.
    J : scalar
        Total Angular Momentum.
    Tc : scalar
        Curie Temperature.

    Returns
    --------
    y : scalar, array
        Magnetic Free Energy
    """

    Ms = Nm * gJ * muB * J  # Saturation magnetization
    lamb = 3 * kB * Tc / (gJ * muB * (J + 1) * Ms)

    def f(sigma, T, B, J, Tc):
        A = np.sinh(3. / (2. * (J + 1.)) *
                    (B / (lamb * Ms) + sigma) * Tc / T)
        B = np.sinh(3. * (2. * J + 1.) / (2. * (J + 1.)) *
                    (B / (lamb * Ms) + sigma) * Tc / T)

        return (sigma**2.) / 2. + (J + 1.) / (3. * J) * T / Tc * np.log(A / B)

    # reduced magnetization of average minimum
    sigma0 = mag.fm_magnetization(T, B, J, gJ, Tc, Nm)

    F0 = free_M(T, B, J, gJ, Tc, Nm)  # average free energy

    return f(sigma, T, B, J, Tc) - f(sigma0, T, B, J, Tc) + F0


# Total Free Energy as a function of Magnetization
def free_total_vs_M(sigma, T, B, J, gJ, Tc, Nm, ThetaD, N, F0):
    """Total free energy as a function of reduced magnetization.

    Parameters
    ----------
    sigma : scalar, 1D array
        Reduced magnetization, between -1 and 1.
    T : scalar
        Temperature.
    B : scalar
        Applied magnetic field.
    J : scalar
        Total angular momentum.
    TC : scalar
        Curie temperature.
    lamb : scalar
        Value of the strength of the parameter of the Molecular Field.
    theta_D : scalar
        Debye temperature.
    F0 : scalar
        Electronic free energy.

    Returns
    --------
    y : scalar, array
        Total free energy.
    """

    fM = free_M_vs_M(sigma, T, B, J, gJ, Tc, Nm)  # magnetic free energy
    fL = free_L(T, ThetaD) * np.ones(np.shape(fM))  # lattice free energy
    F_0 = Nm * (3. * J / (J + 1.) * 1. * kB * Tc) - N * kB * 9. / \
        8. * ThetaD + F0  # offset of free energies to make F start at F0
    return fM * Nm + N * fL + F_0


# Total Free Energy of Stable Phase as a function of Magnetization
def free_total_stable_vs_M(sigma, T, B, J1, J2, gJ, Tc1, Tc2, Nm,
                           ThetaD1, ThetaD2, N, F01, F02, *args):
    """For every magnetization (sigma), computes the minimum between the 2 structures

    Parameters
    ----------
    sigma : array
            Reduced magnetization, from -1 to 1
    T : scalar
            Temperature
    B : scalar
            Applied magnetic field
    args : tuple
            Total free energies of both structures, (F1,F2)

    Returns
    --------
    y : array
            Array containing the values of the minimum free energy.
    """
    if not args:
        F1 = free_total_vs_M(sigma, T, B, J1, gJ, Tc1, Nm, ThetaD1, N, F01)
        F2 = free_total_vs_M(sigma, T, B, J2, gJ, Tc2, Nm, ThetaD2, N, F02)
    else:
#        print '*using args*'
        F1, F2 = args

    return np.minimum(F1, F2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from variables import *

    T = np. arange(300, 320, 2.)
    B = np.arange(0.3159, 0.316, 0.00001)
    
    dsigma = 1e-3
    sigma = np.arange(-1, 1 + dsigma, dsigma)

#    f_m1 = 0 * T
#    f_m2 = 0 * T
#    plt.figure('Magnetic')
#    for b in B:
#        for i, t in enumerate(T):
#            f_m1[i] = free_M(t, b, J, gJ, Tc1, Nm)
#            f_m2[i] = free_M(t, b, J, gJ, Tc2, Nm)
#        plt.plot(T, f_m1, label='1, B=%0.f T' % b)
#        plt.plot(T, f_m2, '--', label='2, B=%0.f T' % b)
#    plt.legend(loc=0, fontsize='xx-small')
#    plt.xlabel('T(K)')
#    plt.ylabel('F(eV/1 moment)')
#
#    plt.figure('Lattice')
#    plt.plot(T, free_L(T, ThetaD1), label='1')
#    plt.plot(T, free_L(T, ThetaD2), label='2')
#    plt.legend(loc=0, fontsize='xx-small')
#    plt.xlabel('T(K)')
#    plt.ylabel('F(eV/1 atom)')
#
#    f_tot1 = 0 * T
#    f_tot2 = 0 * T
#    plt.figure('Total')
#    for b in B:
#        for i, t in enumerate(T):
#            f_tot1[i] = free_total(t, b, J, gJ, Tc1, Nm, ThetaD1, N, DeltaF)
#            f_tot2[i] = free_total(t, b, J, gJ, Tc2, Nm, ThetaD2, N, 0.)
#        plt.plot(T, f_tot1, label='1, B=%0.f T' % b)
#        plt.plot(T, f_tot2, '--', label='2, B=%0.f T' % b)
#    plt.legend(loc=0, fontsize='xx-small')
#    plt.xlabel('T(K)')
#    plt.ylabel('F(eV/cell)')

#    dt = 50  # temperature interval
#    for b in B:
#        plt.figure('free_M_vs_M %.0f T' % b)
#        for i in range(0, len(T), dt):
#            t = T[i]
#            plt.plot(
#                sigma,
#                free_M_vs_M(
#                    sigma,
#                    t,
#                    b,
#                    J,
#                    gJ,
#                    Tc1,
#                    Nm),
#                ':',
#                label='%s, T=%.1f' % (struct1,t))
#        plt.gca().set_color_cycle(None)
#        for i in range(0, len(T), dt):
#            t = T[i]
#            plt.plot(
#                sigma,
#                free_M_vs_M(
#                    sigma,
#                    t,
#                    b,
#                    J,
#                    gJ,
#                    Tc2,
#                    Nm),
#                '--',
#                label='%s, T=%.1f' % (struct2,t))
#        plt.legend(loc=0, fontsize='xx-small')
#        plt.xlabel('Reduced magnetization, $\sigma$')
#        plt.ylabel('F(eV/1 moment)')

    dt = 10  # temperature interval
    for b in B:
        plt.figure('free_total_vs_M %f T' % b)
        for i in range(0, len(T), dt):
            t = T[i]
            plt.plot(
                sigma,
                free_total_vs_M(
                    sigma,
                    t,
                    b,
                    J,
                    gJ,
                    Tc1,
                    Nm,
                    ThetaD1,
                    N,
                    DeltaF),
                ':',
                label='%s, T=%.1f' % (struct1,t))
        plt.gca().set_color_cycle(None)
        for i in range(0, len(T), dt):
            t = T[i]
            plt.plot(
                sigma,
                free_total_vs_M(
                    sigma,
                    t,
                    b,
                    J,
                    gJ,
                    Tc2,
                    Nm,
                    ThetaD2,
                    N,
                    0.),
                '--',
                label='%s, T=%.1f' % (struct2,t))
        plt.gca().set_color_cycle(None)
        for i in range(0, len(T), dt):
            t = T[i]
            f1 = free_total_stable_vs_M(
                    sigma,
                    t,
                    b,
                    J, J,
                    gJ, Tc1,
                    Tc2,
                    Nm, ThetaD1,
                    ThetaD2,
                    N, DeltaF,
                    0.)
            mins = argrelmin(f1)[0]
            maxs = argrelmax(f1)[0]
            print '----------------------'
            print 'T :',t, ', B :',  b
            print 'mins :', mins, 'maxs', maxs
            print 'F:', f1[mins], f1[maxs]
            print 'sigma', sigma[mins], sigma[maxs]
            try:
                print 'DeltaF:', f1[maxs][0] - f1[mins][0], f1[maxs][0] - f1[mins][1]
            except:
                pass
#            print 'DeltaF', f1[maxs] - f1[mins]
            plt.plot(
                sigma,
                f1,
                label='min, T=%.1f' %
                t)
        plt.legend(loc=0, fontsize='xx-small')
        plt.xlabel('Reduced magnetization, $\sigma$')
        plt.ylabel('F(eV/cell)')

    plt.show()