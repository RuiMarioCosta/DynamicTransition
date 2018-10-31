#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:24:31 2017

@author: Rui Costa
"""
import numpy as np
import pickle
from scipy.optimize import fsolve

muB = 5.7883818066 * (10**(-5))  # eV T^-1
kB = 8.6173324 * (10**(-5))  # eV K^-1


class Compound(object):
    def __init__(self, name, J, gJ, Tc, Nm, N, ThetaD):
        self.name = name
        self.J = J
        self.gJ = gJ
        self.Tc = Tc
        self.Nm = Nm
        self.N = N
        self.ThetaD = ThetaD

    def fm_magnetization(self, T, B):
        """Brillouin function. Calculates the reduced magnetization of a ferromagnetic system.

        Parameters
        ----------
        T : scalar, 2D array
            An array with the temperatures.
        B : scalar, 2D array
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
        J = self.J
        gJ = self.gJ
        Tc = self.Tc
        Nm = self.Nm

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

    def database(self):
        try:
            # reload dictionary from file
            database_file = open('DataBase_MCE', 'rb')
            self.db_dict = pickle.load(database_file)
            database_file.close()
        except IOError:
            print 'Creating new database'
            # create file with Gd and Tb compounds if database does not exist
            self.db_dict = {'Gd5Si2Ge2': {'J': 7 / 2., 'gJ': 2., 'TC1': 251., 'TC2': 308., 'DeltaF': 0.36,
                                          'ThetaD1': 250., 'ThetaD2': 278., 'N': 36., 'Nm': 20.},
                            'Tb5Si2Ge2': {'J': 6., 'gJ': 3 / 2., 'TC1': 112., 'TC2': 200., 'DeltaF': 0.11,
                                          'ThetaD1': 153., 'ThetaD2': 170., 'N': 36., 'Nm': 20.}}
            database_file = open('DataBase_MCE', 'wb')
            pickle.dump(self.db_dict, database_file)
            database_file.close()

        self.comboBox_Database.clear()
        self.comboBox_Database.addItems(self.db_dict.keys())


if __name__ == "__main__":
    gd5si2ge2_O = Compound('Gd5Si2Ge2', 7 / 2., 2., 251., )
