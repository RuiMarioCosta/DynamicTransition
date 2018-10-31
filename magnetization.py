# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:24:31 2017

@author: Rui Costa
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import fmin

#import free_energy as free

muB = 5.7883818066 * (10**(-5))  # eV T^-1
kB = 8.6173324 * (10**(-5))  # eV K^-1


#==============================================================================
# Functions
#==============================================================================

def fm_magnetization(T, B, J, gJ, Tc, Nm):
    """Brillouin function. Calculates the reduced magnetization of a ferromagnetic system.

    Parameters
    ----------
    T : scalar
        An array with the temperatures.
    B : scalar
        An array with the magnetic fields.
    J : scalar
        Value of angular momentum.
    Tc : scalar
        Value of Curie temperature.
    gJ : scaclar
        Landé g-factor.
    Nm : scalar
        Number os spins.

    Returns
    --------
    y : scalar, array
        An array with the values of the reduced magnetization.
    """

    Ms = Nm * gJ * muB * J  # Saturation magnetization
    lamb = 3 * kB * Tc / (gJ * muB * (J + 1) * Ms)

    # Function for the computation of sigma
    def B_J(sigma, T, B):
        h = B / (lamb * Ms)
        y = 3. * J / (J + 1.) * (h + sigma) * Tc / T
        return sigma - (2. * J + 1.) / (2. * J * np.tanh((2. * J + 1.)
                                                         * y / (2. * J))) + 1. / (2. * J * np.tanh(y / (2 * J)))

    is_negative = False  # For negative fields
    if B < 0:
        is_negative = True
        B = -1 * B  # Change sign, do calculations por positive magnetic field

    sigma = fsolve(B_J, 0.5, args=(T, B))
    if is_negative:
        sigma = -1. * sigma

    return sigma


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from variables import *

    T = np. arange(1, 400, 1.)
    B = np.arange(0, 5, 1.)
    fm_mag1 = 0. * T
    fm_mag2 = 0. * T

    plt.figure()
    for j, b in enumerate(B):
        for i, t in enumerate(T):
            fm_mag1[i] = fm_magnetization(t, b, J, gJ, Tc1, Nm)
            fm_mag2[i] = fm_magnetization(t, b, J, gJ, Tc2, Nm)

        plt.plot(T, fm_mag1)
        plt.plot(T, fm_mag2, '--')

    plt.show()
