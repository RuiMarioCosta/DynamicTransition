# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:56:30 2017

@author: Rui
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, signal, integrate
from scipy.signal import argrelmin
from scipy.integrate import odeint

import os

def A(T, a0, a1, a2, a3, a4, a5):
    return a0 + a1 * T + a2 * (T**2.) + a3 * (T**3.) + a4 * (T**4.) + a5 * (T**5.)


def B(T, b0, b1, b2, b3, b4, b5):
    return b0 + b1 * T + b2 * (T**2.) + b3 * (T**3.) + b4 * (T**4.) + b5 * (T**5.)


def C(T, c0, c1, c2, c3, c4, c5):
    return c0 + c1 * T + c2 * (T**2.) + c3 * (T**3.) + c4 * (T**4.) + c5 * (T**5.)


def G(M, T, H, a, b, c):
    return 1. / 2. * A(T, *a) * (M**2.) + \
        1. / 4. * B(T, *b) * (M**4.) + \
        1. / 6. * C(T, *c) * (M**6.) - H * M


def dGdM(M, T, H, a, b, c):
    return A(T, *a) * M + B(T, *b) * (M**3.) + C(T, *c) * (M**5.) - H


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


def B_applied(t, dt, Bi, B0, dB, Btype):
    """Applied magnetic field.

    Parameters
    ----------
    t : array
        Time
    dt : scalar
        Time step between jumps
    Bi : scalar
        Initial magnetic field.
    B0 : scalar
        If Btype = 0, maximum applied magnetic field.
        If Btype = 1, final magnetic field.
        If Btype = 2, Heaviside amplitude.
    dB : scalar
        Magnetic field step between jumps
    Btype: Type of magnetic field. 0: triangular shape, 1: right trapezoid.

    Returns
    -------
    y : array
        Triangular function.
    """
    B = np.zeros_like(t) + Bi
    n = 1  # variable used for step increase
    
    if Btype == 0:
        for i in xrange(len(t)):
            if t[i] >= n * dt:
                n += 1
                B[i:] += dB
                if B0 <= np.abs(B[i]):
                    dB = -dB
    elif Btype == 1:
        for i in xrange(len(t)):
            if B[i] >= B0:
                break
            elif t[i] > n * dt:
                n += 1
                B[i:] += dB
    elif Btype == 2:
        B[np.where(t > len(t)*dt/4)[0]] = B0
    elif Btype == 3:
        for i in xrange(len(t)):
            if t[i] >= n * dt:
                n += 1
                B[i:] += dB
                if B[i] >= B0 or B[i] <= 0:
                    dB = -dB
    return B


def dMdt(M, t, v, T, H, a, b, c):
#    print M.shape, T.shape, H.shape
    dmdt = -v * dGdM(M, T, H, a, b, c)
#    print dmdt.shape
    return dmdt


def evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
              Hi, H0, dH, dt_H, Htype):

    T = T_applied(t, dt_T, T_min, T_max, Ti, dT)  # temperature
    H = B_applied(t, dt_H, Hi, H0, dH, Htype)  # magnetic field
    mag_evolution = np.zeros_like(t)  # magnetization variable

    # calculate initial magnetization
    abs_min0 = np.argmin(G(M, T[0], H[0], a, b, c))
    if H[0] == 0.:  # if H=0, magnetization goes to the negative state
        mag_evolution[0] = -M[abs_min0]  # demand that it is positive
    else:
        mag_evolution[0] = M[abs_min0]
        
    # determine following magnetizations
    for i in xrange(0, len(t) - 1):
#        sol = odeint(dMdt, mag_evolution[i], t[i:i+2], args=(v, T[i], H[i], a, b, c))
#        print i, t[i], sol[0], sol[1]
#        mag_evolution[i+1] = sol[1]
        mag_evolution[i+1] = np.argmin(G(M, T[i], H[i], a, b, c))
    return mag_evolution


def M_Tt(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T, 
         Hi, H0, dH, dt_H, Htype):
    
    nT = int(round((T_max - T_min) / dT))  # number of temperatures
    mag = np.zeros((nT, len(t)))  # magnetization variable
    Ti = T_min # inicial temperature
    
    # calculate the magnetization for the various temperatures
    for i in range(nT):
        print '----M_Tt----', i, nT, Ti, T_min, T_max
        mag[i] = evolution(compound, M, dm, a, b, c, v, t, dt, T_min,
                           T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype)
        Ti += dT        
    return mag

def dMdT_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
                   Hi, H0, dH, dt_H, Htype):
    mag = M_Tt(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
               Hi, H0, dH, dt_H, Htype)
    dMdT, dMdt = np.gradient(mag, dT, dt)  # compute gradient
    return dMdT


def entropy(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
            Hi, H0, dH, dt_H, Htype):
    
    dMdT = dMdT_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti,
                          dT, dt_T, Hi, H0, dH, dt_H, Htype)
    print 1, dMdT.shape
#    nT = int((T_max - T_min) / dT)  # number of temperatures
#    N_pieces = len(max_indices) + len(min_indices) + 1
#    ent = np.zeros((N_pts, N_pieces))
    
    # H needs to be partitioned otherwise there are multiple magnetizations
    # for the same magnetic field
    dH_interval = int(dt_H / dt)
    H = B_applied(t, dt_H, Hi, H0, dH, Htype)[::dH_interval]  # magnetic field
    dMdT = dMdT[:, ::dH_interval]
    max_indices = signal.argrelmax(H)[0]
    min_indices = signal.argrelmin(H)[0]
    nT = int(round((T_max - T_min) / dT))  # number of temperatures
    Ti = T_min # inicial temperature

    print 2, dMdT.shape, H.shape, max_indices, min_indices, nT
    
    # the integration is done in pieces, i.e., the first goes from H=0 to H=H0,
    # resulting in the entropy change, S(H0) - S(0), the next one goes from
    # H=H0 to H=-H0 resulting in S(-H0) - S(H0), and so on until there are no
    # more.
    if Htype == 0:
        N_pieces = 6 #len(max_indices) + len(min_indices) + 1
        ent = np.zeros((nT, N_pieces+2))
    
        # create variable that keeps limits of integration
        lim_int = np.zeros(N_pieces + 1, int)
        for i in xrange(len(max_indices)):
            lim_int[4 * i + 1] = max_indices[i]
        for i in xrange(len(min_indices)):
            lim_int[4 * i + 3] = min_indices[i]
        lim_int[::2] = np.where(H == 0)[0] # insert indices of zeros
    
    elif Htype == 3:
        N_pieces = 3 # if Htype = 3
        ent = np.zeros((nT, N_pieces))
        
        # create variable that keeps limits of integration
        lim_int = np.zeros(N_pieces + 1, int)
        lim_int[-1] = -1
        lim_int[1] = max_indices
        lim_int[2] = min_indices
                
    print 3, ent.shape, lim_int
    for i in xrange(nT): # for each temperature
        for j in xrange(N_pieces): # integrate each piece
            print j, 'H', H[lim_int[j]], H[lim_int[j+1]]
            ent[i, j] = integrate.simps(dMdT[i, lim_int[j]:lim_int[j + 1]],
                                        H[lim_int[j]:lim_int[j + 1]])
        
        if Htype == 0:
            ent[i,-2] = integrate.simps(dMdT[i, lim_int[1]:lim_int[3]],
                                            H[lim_int[1]:lim_int[3]])
            ent[i,-1] = integrate.simps(dMdT[i, lim_int[3]:lim_int[5]],
                                            H[lim_int[3]:lim_int[5]])
        print '----entropy----', i, Ti, nT
        Ti += dT

    return ent  # integrate to obtain entropy

def entropy_equilibrium(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
            Hi, H0, dH, dt_H, Htype):

    T = T_applied(t, dt_T, T_min, T_max, Ti, dT)

    G0  = np.zeros_like(T)
    Gh1 = np.zeros_like(T)
    Gh2 = np.zeros_like(T)
    Gh3 = np.zeros_like(T)
    
    for i in xrange(len(T)):
        M0  = M[np.argmin(G(M, T[i], 1e-5, a, b, c))]
        Mh1 = M[np.argmin(G(M, T[i], 5000., a, b, c))]
        Mh2 = M[np.argmin(G(M, T[i], 10000., a, b, c))]
        Mh3 = M[np.argmin(G(M, T[i], 15000., a, b, c))]
        print M0, Mh1, Mh2, Mh3
        G0[i]  = G(M0, T[i], 1e-5, a, b, c)
        Gh1[i] = G(Mh1, T[i], 5000., a, b, c)
        Gh2[i] = G(Mh2, T[i], 10000., a, b, c)
        Gh3[i] = G(Mh3, T[i], 15000., a, b, c)
    
    print G0[:10], G0.shape, G0[-10:]
    
    S0  = -np.gradient(G0, dT)
    Sh1 = -np.gradient(Gh1, dT)
    Sh2 = -np.gradient(Gh2, dT)
    Sh3 = -np.gradient(Gh3, dT)

    plt.figure('G, plot_entropy_equilibrium')
    plt.plot(T, G0, label='0')
    plt.plot(T, Gh1, label='h1')
    plt.plot(T, Gh2, label='h2')
    plt.plot(T, Gh3, label='h3')
    
    plt.figure('S, plot_entropy_equilibrium')
    plt.plot(T, S0, label='0')
    plt.plot(T, Sh1, label='h1')
    plt.plot(T, Sh2, label='h2')
    plt.plot(T, Sh3, label='h3')
    
    plt.figure('DS, plot_entropy_equilibrium')
    plt.plot(T, Sh1 - S0, label='h1')
    plt.plot(T, Sh2 - S0, label='h2')
    plt.plot(T, Sh3 - S0, label='h3')

    plt.legend(loc=0)
    plt.show()
    
    if 1 == 1: # save to .txt
        file_name = ('entropy-eq, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
                                 'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g).txt')\
                                 % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
                                    Hi, H0, dH, dt_H, Htype)
        # save data to .txt
        file_data = np.vstack((T, S0, Sh1, Sh2, Sh3)).T # save data to .txt
        np.savetxt('data-txt\\' + file_name, file_data, delimiter=' ')
    return None


def plot_G(M, Ti, Tf, dT, Hi, Hf, dH, a, b, c):
    for i in range(Hi, Hf, dH):
        plt.figure('plot_G, H=%g' % H[i])
        for j in range(Ti, Tf, dT):
            plt.plot(M, G(M, T[j], H[i], a, b, c), label='T=%g' % T[j])
            plt.legend(loc=0, fontsize='xx-small')
        plt.grid()
        plt.ylabel('G(emu G/g, erg/g)')
        plt.xlabel('M (emu/g)')
    plt.show()
    print 'plot_G'

def plot_dGdM(M, Ti, Tf, dT, Hi, Hf, dH, a, b, c):
    for i in range(Hi, Hf, dH):
        plt.figure('plot_dGdM, H=%g' % H[i])
        for j in range(Ti, Tf, dT):
            plt.plot(M, dGdM(M, T[j], H[i], a, b, c), label='T=%g' % T[j])
            plt.legend(loc=0, fontsize='xx-small')
        plt.grid()
        plt.ylabel('dGdM')
        plt.xlabel('M (emu/g)')
    plt.show()
    print 'plot_dGdM'
    
def plot_T(t, dt, T_min, T_max, Ti, dT):
    print 'plot_T'
    plt.figure('Temperature')
    plt.plot(t, T_applied(t, dt, T_min, T_max, Ti, dT), '-o', ms=3.)
    plt.xlabel('t')
    plt.ylabel('T(K)')
    plt.grid()
    plt.title('v = %g Hz, dT/dt = %g K/s, dH/dt = %g Oe/s' % (v, dT / dt_T, dH / dt_H))

    plt.show()


def plot_H(t, dt, Hi, H0, dH, Htype):
    print 'plot_H'
    plt.figure('Magnetic field, v = %g, dTdt = %g, dHdt = %g' %
               (v, dT / dt_T, dH / dt_H))
    plt.plot(t, B_applied(t, dt, Hi, H0, dH, Htype), '-o', ms=3.)
    plt.xlabel('t')
    plt.ylabel('H(Oe)')
    plt.grid()
    plt.title('v = %g Hz, dT/dt = %g K/s, dH/dt = %g Oe/s, Htype = %g' %
              (v, dT / dt_T, dH / dt_H, Htype))

    plt.show()

def plot_evolution2(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
              Hi, H0, dH, dt_H, Htype):
    print 'plot_evolution'
    plt.figure('plot_evolution, v = %g, dTdt = %g, dHdt = %g' %
               (v, dT / dt_T, dH / dt_H))
    for v in [1., 0.1, 0.01, 0.001, 0.0005]:
        plt.plot(t, evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
                  Hi, H0, dH, dt_H, Htype), '-o', ms=3., label='v=%g'%v)
    plt.xlabel('t')
    plt.ylabel('M(emu/g)')
    plt.grid()
    plt.title('v = %g Hz, dT/dt = %g K/s, dH/dt = %g Oe/s, Htype = %g' %
              (v, dT / dt_T, dH / dt_H, Htype))
    plt.legend(loc=0)

    plt.show()

def plot_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                   dt_T, Hi, H0, dH, dt_H, Htype,
                   fig=None, axes=None):
    
    Mag = evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
                  Hi, H0, dH, dt_H, Htype)
    
    if fig is None:
        fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(nrows = 2, ncols=3,
             num='plot_evolution')
        fig.set_size_inches(10, 6)
        fig.tight_layout()
    else:
        ((ax1,ax2,ax3), (ax4,ax5,ax6)) = axes
        
    # plot graphs
    T = T_applied(t, dt_T, T_min, T_max, Ti, dT)
    H = B_applied(t, dt_H, Hi, H0, dH, Htype)
    
    ax1.plot(t, T)
    ax1.set(xlabel='t', ylabel='T(K)')
    ax1.grid()
    
    ax2.set_title(('compound %g, v = %g Hz, dT/dt = %g K/s, dH/dt = %g Oe/s, '
                  'Htype = %g') %
            (compound, v, dT/dt_T, dH/dt_H, Htype))
    if Htype == 1:
        t_shift = H0/dH*dt_H
    else:
        t_shift = 0
    ax2.plot(t-t_shift, H)
    ax2.set(xlabel='t', ylabel='H(Oe)')
    ax2.grid()

    ax3.plot(t-t_shift, Mag)
    if plot_evolutions.has_been_called in [0, 1]:
        if compound == 3:
            m_18mTs_8000Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-8000Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_9000Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-9000Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_7200Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-7200Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_9800Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-9800Oe-Mt.txt',
                                       delimiter='\t')
            ax3.plot(m_18mTs_7200Oe_Mt[:, 0], m_18mTs_7200Oe_Mt[:, 1], 'ok',
                     label='Lovell 18 mT/s, 072 T', ms=1)
            ax3.plot(m_18mTs_8000Oe_Mt[:, 0], m_18mTs_8000Oe_Mt[:, 1], 'or',
                     label='Lovell 18 mT/s, 0.8 T', ms=1)
            ax3.plot(m_18mTs_9000Oe_Mt[:, 0], m_18mTs_9000Oe_Mt[:, 1], 'og',
                     label='Lovell 18 mT/s, 0.9 T', ms=1)
            ax3.plot(m_18mTs_9800Oe_Mt[:, 0], m_18mTs_9800Oe_Mt[:, 1], 'ob',
                     label='Lovell 18 mT/s, 0.98 T', ms=1)
        elif compound == 6:
            if Ti == 192.5:
                m_t1 = np.loadtxt('Lovell-Mt-192.5K-0,33T-sample2.txt',
                                  delimiter='\t')
                m_t2 = np.loadtxt('Lovell-Mt-192.5K-0,47T-sample2.txt',
                                  delimiter='\t')
                m_t3 = np.loadtxt('Lovell-Mt-192.5K-0,61T-sample2.txt',
                                  delimiter='\t')
                label1 = 'Lovell, 8.3 mT/s, 192.5K, 0.33 T'
                label2 = 'Lovell, 8.3 mT/s, 192.5K, 0.47 T'
                label3 = 'Lovell, 8.3 mT/s, 192.5K, 0.61 T'
            elif Ti == 193.5:
                m_t1 = np.loadtxt('Lovell-MH-193.5K-sample2.txt',
                                  delimiter='\t')
            elif Ti == 194.5:
                m_t1 = np.loadtxt('Lovell-Mt-194.5K-0,74T-sample2.txt',
                                  delimiter='\t')
                m_t2 = np.loadtxt('Lovell-Mt-194.5K-0,90T-sample2.txt',
                                  delimiter='\t')
                m_t3 = np.loadtxt('Lovell-Mt-194.5K-0,99T-sample2.txt',
                                  delimiter='\t')
                label1 = 'Lovell, 8.3 mT/s, 194.5K, 0.74 T'
                label2 = 'Lovell, 8.3 mT/s, 194.5K, 0.90 T'
                label3 = 'Lovell, 8.3 mT/s, 194.5K, 0.99 T'
                
            ax3.plot(m_t1[:, 0], m_t1[:, 1], 'oC1', label=label1, ms=1)
            ax3.plot(m_t2[:, 0], m_t2[:, 1], 'oC2', label=label2, ms=1)
            ax3.plot(m_t3[:, 0], m_t3[:, 1], 'oC3', label=label3, ms=1)
    ax3.set(xlabel='t', ylabel='M')
    ax3.grid()
    ax3.legend(loc=0, fontsize='xx-small')
    
    ax6.plot(t-t_shift, Mag)
    if plot_evolutions.has_been_called in [0, 1]:
        if compound == 3:
            m_18mTs_7200Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-7200Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_7400Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-7400Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_7600Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-7600Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_7800Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-7800Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_8000Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-8000Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_9000Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-9000Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_9800Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-9800Oe-Mt.txt',
                                       delimiter='\t')
            ax6.plot(m_18mTs_7200Oe_Mt[:, 0], m_18mTs_7200Oe_Mt[:, 1], 'oC1',
                     label='Lovell 18 mT/s, 0.72 T', ms=1)
            ax6.plot(m_18mTs_7400Oe_Mt[:, 0], m_18mTs_7400Oe_Mt[:, 1], 'oC2',
                     label='Lovell 18 mT/s, 0.74 T', ms=1)
            ax6.plot(m_18mTs_7600Oe_Mt[:, 0], m_18mTs_7600Oe_Mt[:, 1], 'oC3',
                     label='Lovell 18 mT/s, 0.76 T', ms=1)
            ax6.plot(m_18mTs_7800Oe_Mt[:, 0], m_18mTs_7800Oe_Mt[:, 1], 'oC4',
                     label='Lovell 18 mT/s, 0.78 T', ms=1)
            ax6.plot(m_18mTs_8000Oe_Mt[:, 0], m_18mTs_8000Oe_Mt[:, 1], 'oC5',
                     label='Lovell 18 mT/s, 0.8 T', ms=1)
            ax6.plot(m_18mTs_9000Oe_Mt[:, 0], m_18mTs_9000Oe_Mt[:, 1], 'oC6',
                     label='Lovell 18 mT/s, 0.9 T', ms=1)
            ax6.plot(m_18mTs_9800Oe_Mt[:, 0], m_18mTs_9800Oe_Mt[:, 1], 'oC7',
                     label='Lovell 18 mT/s, 0.98 T', ms=1)
        elif compound == 6:
            if Ti == 192.5:
                m_t1 = np.loadtxt('Lovell-Mt-192.5K-0,33T-sample2.txt',
                                  delimiter='\t')
                m_t2 = np.loadtxt('Lovell-Mt-192.5K-0,47T-sample2.txt',
                                  delimiter='\t')
                m_t3 = np.loadtxt('Lovell-Mt-192.5K-0,61T-sample2.txt',
                                  delimiter='\t')
                label1 = 'Lovell, 8.3 mT/s, 192.5K, 0.33 T'
                label2 = 'Lovell, 8.3 mT/s, 192.5K, 0.47 T'
                label3 = 'Lovell, 8.3 mT/s, 192.5K, 0.61 T'
            elif Ti == 193.5:
                m_t1 = np.loadtxt('Lovell-MH-193.5K-sample2.txt',
                                  delimiter='\t')
            elif Ti == 194.5:
                m_t1 = np.loadtxt('Lovell-Mt-194.5K-0,74T-sample2.txt',
                                  delimiter='\t')
                m_t2 = np.loadtxt('Lovell-Mt-194.5K-0,90T-sample2.txt',
                                  delimiter='\t')
                m_t3 = np.loadtxt('Lovell-Mt-194.5K-0,99T-sample2.txt',
                                  delimiter='\t')
                label1 = 'Lovell, 8.3 mT/s, 194.5K, 0.74 T'
                label2 = 'Lovell, 8.3 mT/s, 194.5K, 0.90 T'
                label3 = 'Lovell, 8.3 mT/s, 194.5K, 0.99 T'
                
            ax6.plot(m_t1[:, 0], m_t1[:, 1], 'oC1', label=label1, ms=1)
            ax6.plot(m_t2[:, 0], m_t2[:, 1], 'oC2', label=label2, ms=1)
            ax6.plot(m_t3[:, 0], m_t3[:, 1], 'oC3', label=label3, ms=1)
    ax6.set(xlabel='t', ylabel='M')
    ax6.grid()
    ax6.set_xlim(-50, 50)
#    ax6.set_ylim(0,120)
    ax6.legend(loc=0, fontsize='xx-small')

    ax4.plot(H, Mag)
    if plot_evolutions.has_been_called in [0, 1]:
        if compound == 3:
            m_1mTs_Lovell = np.loadtxt('LaFe11.6Si1.4-Lovell-1mTs.txt',
                                       delimiter='\t')
            m_18mTs_Lovell = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs.txt',
                                        delimiter='\t')
            m_18mTs_8000Oe_Lovell = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-8000Oe.txt',
                                               delimiter='\t')
            m_18mTs_9000Oe_Lovell = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-9000Oe.txt',
                                               delimiter='\t')
        
            ax4.plot(m_1mTs_Lovell[:, 0], m_1mTs_Lovell[:, 1], 'ob',
                     label='Lovell 1 mT/s', ms=1)
            ax4.plot(m_18mTs_Lovell[:, 0], m_18mTs_Lovell[:,1], 'or',
                     label='Lovell 18 mT/s', ms=1)
            ax4.plot(m_18mTs_8000Oe_Lovell[:, 0], m_18mTs_8000Oe_Lovell[:,1], 'oc',
                     label='Lovell 18 mT/s, 0.8 T', ms=1)
            ax4.plot(m_18mTs_9000Oe_Lovell[:, 0], m_18mTs_9000Oe_Lovell[:,1], 'og',
                     label='Lovell 18 mT/s, 0.9 T', ms=1)
        elif compound == 6:
            m_H_192K = np.loadtxt('Lovell-MH-192.5K-sample2.txt',
                                       delimiter='\t')
            m_H_193K = np.loadtxt('Lovell-MH-193.5K-sample2.txt',
                                       delimiter='\t')
            m_H_194K = np.loadtxt('Lovell-MH-194.5K-sample2.txt',
                                       delimiter='\t')
            m_H_195K = np.loadtxt('Lovell-MH-195.5K-sample2.txt',
                                       delimiter='\t')
            m_H_196K = np.loadtxt('Lovell-MH-196.5K-sample2.txt',
                                       delimiter='\t')
            ax4.plot(m_H_192K[:, 0], m_H_192K[:, 1], 'oC1',
                     label='8.3 mT/s, 192.5 K, 0.47 T', ms=1)
            ax4.plot(m_H_193K[:, 0], m_H_193K[:, 1], 'oC2',
                     label='8.3 mT/s, 193.5 K, 0.66 T', ms=1)
            ax4.plot(m_H_194K[:, 0], m_H_194K[:, 1], 'oC3',
                     label='8.3 mT/s, 194.5 K, 0.90 T', ms=1)
            ax4.plot(m_H_195K[:, 0], m_H_195K[:, 1], 'oC4',
                     label='8.3 mT/s, 195.5 K, 1.10 T', ms=1)
            ax4.plot(m_H_196K[:, 0], m_H_196K[:, 1], 'oC5',
                     label='8.3 mT/s, 196.5 K, 1.30 T', ms=1)
    ax4.set(xlabel='H(Oe)', ylabel='M')
    ax4.grid()
    ax4.legend(loc=0, fontsize='xx-small')

    ax5.plot(T, Mag)
    ax5.set(xlabel='T(K)', ylabel='M')
    ax5.grid()

    plt.show()
#    print 'plot_evolution'
    
    
    for i in xrange(len(Mag)):
#        if np.abs(Mag[i] - Mag[-1]) < dm:
        if Mag[i] > 0.9*Mag[-1]: # if magnetization is at least 90% saturation
            print Ti, i, t[i], Mag[i]
            break
    
    if 1 == 1: # save
        file_name = ('evolution, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
                     'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(3).png')\
                     % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
                        Hi, H0, dH, dt_H, Htype)
        
        # save data to .txt
        print '...saving txt - plot_evolution'
        file_data = np.vstack((t, T, H, Mag)).T # save data to .txt
        np.savetxt('data-txt/' + file_name[:-3]+'txt', file_data, delimiter=' ')
        
        # save figure
        if plot_evolutions.has_been_called == -1 or plot_evolutions.has_been_called == 0:  # if last called
            if plot_evolutions.has_been_called == -1: # name: evolutions
                file_name = ('evolutions, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
                             'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(3).png')\
                             % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
                                Hi, H0, dH, dt_H, Htype)
    
            # directory's path and file name being executed
            dirname, main_filename = os.path.split(os.path.abspath(__file__))
            # if file does not exists in data folder, do calculations
            if find(file_name, dirname + '\\results') == None:
                print '...saving png - plot_evolution'
                fig.savefig('results/' + file_name, bbox_inches='tight', dpi=300)
            else:
                print file_name, '--File already saved.'


def plot_evolutions(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                    dt_T, Hi, H0, dH, dt_H, Htype):
    plot_evolutions.has_been_called = 0
    nT = int((T_max - T_min) / dT)
    Ti = T_min
    n = 1
    list_range = [i for i in xrange(0, nT, n)]
    fig, axes = plt.subplots(nrows = 2, ncols=3, num='plot_evolutions')
    fig.set_size_inches(10, 6)
#    fig.tight_layout()
    for i in list_range:
        plot_evolutions.has_been_called += 1
        if i == list_range[-1]:
            plot_evolutions.has_been_called = -1
#        print '----', i, nT, Ti
        plot_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti,
                       dT, dt_T, Hi, H0, dH, dt_H, Htype, fig, axes)
        Ti += dT * n
    print 'plot_evolutions'
    

def plot_M_T(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
             Hi, H0, dH, dt_H, Htype):
    mag = M_Tt(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti,
                        dT, dt_T, Hi, H0, dH, dt_H, Htype)
    
    H = B_applied(t, dt_H, Hi, H0, dH, Htype)  # magnetic field
    Tvar = np.arange(T_min, T_max, dT)
    # M vs T
    plt.figure('M-T1')
    for i in xrange(mag.shape[1]/3+1):
        print 1, i, t[i], H[i]
        plt.plot(Tvar, mag[:,i], label='H=%g' % H[i])
    plt.legend(loc=0)
    plt.figure('M-T2')
    for i in xrange(mag.shape[1]/3, mag.shape[1]*2/3+1):
        print 2, i, t[i], H[i]
        plt.plot(Tvar, mag[:,i], label='H=%g' % H[i])
    plt.legend(loc=0)
    plt.figure('M-T3')
    for i in xrange(mag.shape[1]*2/3, mag.shape[1]):
        print 3, i, t[i], H[i]
        plt.plot(Tvar, mag[:,i], label='H=%g' % H[i])
    plt.legend(loc=0)
    
    # M vs H
    print len(Tvar)
    plt.figure('M-H1')
    for i in xrange(len(Tvar)):
        print 1, i, Tvar[i]
        plt.plot(H[:mag.shape[1]/3+1],
                 mag[i, 0:mag.shape[1]/3+1],
                 label='T=%g' % Tvar[i])
    plt.legend(loc=0)
    plt.figure('M-H2')
    for i in xrange(len(Tvar)):
        print 2, i, Tvar[i]
        plt.plot(H[mag.shape[1]/3:mag.shape[1]*2/3+1],
                 mag[i, mag.shape[1]/3:mag.shape[1]*2/3+1],
                 label='T=%g' % Tvar[i])
    plt.legend(loc=0)
    plt.figure('M-H3')
    for i in xrange(len(Tvar)):
        print 3, i, Tvar[i]
        plt.plot(H[mag.shape[1]*2/3:-1],
                 mag[i, mag.shape[1]*2/3:-1],
                 label='T=%g' % Tvar[i])
    plt.legend(loc=0)
    plt.figure('M-H')
    for i in xrange(len(Tvar)):
        print 3, i, Tvar[i]
        plt.plot(H,
                 mag[i],
                 label='T=%g' % Tvar[i])
    plt.legend(loc=0)
    plt.show()
    
    
    print mag.shape, mag.shape[1]/3, mag.shape[1]*2//3, mag.shape[1]
    print t[mag.shape[1]/3],t[mag.shape[1]*2/3], t[mag.shape[1]-1]
    print H[mag.shape[1]/3],H[mag.shape[1]*2/3], H[mag.shape[1]-1]
    


def plot_dMdT_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti,
                        dT, dt_T, Hi, H0, dH, dt_H, Htype):
    dMdT = dMdT_evolution(compound, M, dm, a, b, c, v, t, dt, T_min,
                               T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype)
    
    plt.figure('plot_dMdT_evolution_2D')
    im = plt.imshow(dMdT,
                    extent=[t[0], t[-1], T_min, T_max],
                    aspect='auto',
                    origin='lower')
    plt.colorbar(im)
    plt.xlabel('t')
    plt.ylabel('T(K)')
    plt.title('compound=%g, v = %g, dT/dt = %g, dH/dt = %g' %
              (compound, v, dT / dt_T, dH / dt_H))

    plt.show()
    
    #################################
    
    dH_interval = int(dt_H / dt)
    H = B_applied(t, dt_H, Hi, H0, dH, Htype)[::dH_interval]  # magnetic field
    dMdT = dMdT[:, ::dH_interval]
    max_indices = signal.argrelmax(H)[0]
    min_indices = signal.argrelmin(H)[0]
    nT = int(round((T_max - T_min) / dT))  # number of temperatures
    Ti = T_min # inicial temperature
    
    N_pieces = 6 #len(max_indices) + len(min_indices) + 1

    # create variable that keeps limits of integration
    lim_int = np.zeros(N_pieces + 1, int)
    for i in xrange(len(max_indices)):
        lim_int[4 * i + 1] = max_indices[i]
    for i in xrange(len(min_indices)):
        lim_int[4 * i + 3] = min_indices[i]
    lim_int[::2] = np.where(H == 0)[0] # insert indices of zeros
    
    for i in xrange(nT): # for each temperature
        plt.figure('plot_dMdT_evolution i=%g, Ti=%g' % (i, Ti))
        for j in xrange(N_pieces): # integrate each piece
            print j, 'H', H[lim_int[j]], H[lim_int[j+1]]
            plt.plot(H[lim_int[j]:lim_int[j + 1]],
                     dMdT[i, lim_int[j]:lim_int[j + 1]],
                     label='T=%g, j=%g' % (Ti, j),
                     lw = 8-j)
        Ti += dT
        plt.grid()
        plt.legend(loc=0)
    
    print 'plot_dMdT_evolution'


def plot_entropy(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                    dt_T, Hi, H0, dH, dt_H, Htype):
    
    ent = entropy(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                  dt_T, Hi, H0, dH, dt_H, Htype)
    
#    T = np.arange(T_min, T_max, dT)
    T = np.linspace(T_min, T_max, int((T_max-T_min)/dT), endpoint=False)
    print ent.shape, T.shape
    
    if 1 == 1: # save to .txt
        file_name = ('entropy, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
                                 'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(3).txt')\
                                 % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
                                    Hi, H0, dH, dt_H, Htype)
        print T.shape, ent.shape
        # save data to .txt
        file_data = np.vstack((T, ent.T)).T # save data to .txt
        np.savetxt('data-txt\\' + file_name, file_data, delimiter=' ')
    
    # plot figure
    plt.figure('plot_entropy, compound=%g, v = %g, dTdt = %g, dHdt = %g, dT = %g, dH = %g, dt = %g' % (
        compound, v, dT / dt_T, dH / dt_H, dT, dH, dt), figsize=(10, 6))
    for i in xrange(len(ent[0])):
        plt.plot(T, ent[:, i], label='%d' % i, marker='o', ms=10-i)

    plt.legend(loc=0)
    plt.xlabel('T(K)')
    plt.ylabel('S')
    plt.title('compound = %g, v = %g $s^{-1}$, H0 = %g Oe \n dT = %g K, dH = %g Oe, dt = %g s, dT/dt = %g K/s, dH/dt = %g Oe/s' %
              (compound, v, H0, dT, dH, dt, dT / dt_T, dH / dt_H))

    plt.show()
    
#    if 1 == 1:
#        file_name = ('entropy, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
#                                 'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(3).txt')\
#                                 % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
#                                    Hi, H0, dH, dt_H, Htype)
#        print T.shape, ent.shape
#        # save data to .txt
#        file_data = np.vstack((T, ent.T)).T # save data to .txt
#        np.savetxt('data-txt\\' + file_name, file_data, delimiter=' ')
#
#    print 'plot_entropy'

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
        
        
if __name__ == '__main__':
    compound = 3
    if compound == 1:  # LaFe11Si2, SOPT, Tc = 250 K, 150 - 300 K, Ms =
        c = [0, 0, 0, 0, 0, 0]
        a = [-3.51683152e+04, 4.87403171e+02, -2.23828730e+00,
             1.35479224e-03, 1.55205820e-05, -2.99854199e-08]
        b = [-1.63425257e+00, 5.48282863e-02, -6.30962333e-04,
             3.41585145e-06, -8.90087664e-09, 9.04181075e-12]
    elif compound == 2:  # LaFe11.4Si1.6, FOPT, Tc = 208 K, 200 - 230 K, Ms =
        a = [8.68457668e+07, -1.91975049e+06, 1.69676748e+04, -
             7.49544217e+01, 1.65490805e-01, -1.46097415e-04]
        b = [6.92456427e+02, -2.51900616e+01, 3.13238281e-01, -
             1.79886193e-03, 4.92237726e-06, -5.21587610e-09]
        c = [-2.78865095e-01, 6.67283924e-03, -6.36294361e-05,
             3.02349528e-07, -7.16135249e-10, 6.76568184e-13]
    elif compound == 3:  # LaFe11.6Si1.4, FOPT, Tc = 191 K, T = 194.5 K, Ms = 120 g/emu
        #        a = [239.06613, 0, 0, 0, 0, 0]
        a = [-13046.180237142858, 68.30460857142857, 0, 0, 0, 0]
        b = [-0.04248, 0, 0, 0, 0, 0]
        c = [2.4083e-6, 0, 0, 0, 0, 0]

        dm = 1e-2
        dt = 1e0
        dh = 1e3
        M = np.arange(-200, 200 + dm, dm)
        T = np.arange(190, 200. + dt, dt)
        H = np.arange(0., 15000. + dh, dh)

        Ti_G = 0  # starting index
        Tf_G = len(T) - int(0 / dt)  # ending index
        dT_G = int(1 / dt)  # index step
        Hi_G = 0  # starting magnetic field
        Hf_G = len(H)  # ending magnetic field
        dH_G = int(5000 / dh)  # index step

        v = 0.004  # attempt frequency

        dt = 0.1
        t = np.arange(0, 25 + dt, dt)

        dt_T = dt
        dT = 0.01/.2
        T_min = 188. #190.
        T_max = 202. #200.
        Ti = 188. #194.5

        Hi = 0.
        H0 = 10000.
        dt_H = dt # dt_H <= dt
        dH = 5.  # 10 Oe = 1 mT

    elif compound == 4:  # Amaral, Tc = 195 K, Ms = 100 emu/g, FOPT
        a = [-4500., 25, 0, 0, 0, 0]
        b = [-0.18, 0, 0, 0, 0, 0]
        c = [2.33e-5, 0, 0, 0, 0, 0]

        dm = 1e-2
        dt = 1e0
        dh = 1e2
        M = np.arange(-200,200 + dm, dm)
        T = np.arange(185, 220. + dt, dt)
        H = np.arange(0., 20000. + dh, dh)

        Ti_G = 0  # starting index
        Tf_G = len(T) - int(0 / dt)  # ending index
        dT_G = int(10 / dt)  # index step
        Hi_G = 0  # starting magnetic field
        Hf_G = len(H)  # ending magnetic field
        dH_G = int(10000 / dh)  # index step

        v = 0.001  # attempt frequency

        dt = 0.1
        t = np.arange(0, 72 + dt, dt)

        T_min = 170.
        T_max = 240.
        Ti = 170.
        dt_T = dt
        dT = 0.1

        Hi = 0.
        H0 = 15000.
        dt_H = dt # dt <= dt_H
        dH = 400.  # 10 Oe = 1 mT
    elif compound == 5:  #Tc = 631 K, Ms = 15000 emu/g, SOPT
        a = [-0.007320862, 0.000011602, 0, 0, 0, 0]
        b = [2.1028e-8, 0, 0, 0, 0, 0]
        c = [0, 0, 0, 0, 0, 0]

        dM = 1e0
        dT = 1e0
        dH = 1e2
        M = np.arange(-20000, 20000 + dM, dM)
        T = np.arange(600, 650. + dT, dT)
        H = np.arange(0., 20000. + dH, dH)

        Ti_G = 0  # starting index
        Tf_G = len(T) - int(0 / dT)  # ending index
        dT_G = int(10 / dT)  # index step
        Hi_G = 0  # starting magnetic field
        Hf_G = len(H)  # ending magnetic field
        dH_G = int(10000 / dH)  # index step

        v = 1.  # attempt frequency

        dt = 0.1
        t = np.arange(0, 50. + dt, dt)

        T_min = 170.
        T_max = 230.
        Ti = 170.
        dt_T = np.inf
        dT = 0.5

        Hi = 0.
        H0 = 15000.
        dt_H = 1.
        dH = 100.  # 10 Oe = 1 mT
    elif compound == 6:
        # LaFe11.6Si1.4, FOPT, Tc = 191 K, T = 194.5 K, Ms = 120 g/emu
        a = [-9.48643593e6, 1.46035031e5, -7.49494482e2, 1.28248083, 0, 0]
        b = [6.20559722e3, -9.57604139e1, 4.92543380e-1, -8.44427354e-4, 0, 0]
        c = [-4.26317539e-1, 6.58106113e-3, -3.38616203e-5, 5.80725973e-8, 0, 0]

        dm = 1e0
        dt = 1e0
        dh = 1e3
        M = np.arange(-200, 200 + dm, dm)
        T = np.arange(192.5, 196.5 + dt, dt)
        H = np.arange(0., 30000. + dh, dh)

        Ti_G = 0  # starting index
        Tf_G = len(T) - int(0 / dt)  # ending index
        dT_G = int(1 / dt)  # index step
        Hi_G = 0  # starting magnetic field
        Hf_G = len(H)  # ending magnetic field
        dH_G = int(10000 / dh)  # index step

        v = 0.1  # attempt frequency

        dt = 1.
        t = np.arange(0, 2400 + dt, dt) # 2400 / 400

        T_min = 195.9
        T_max = 196.7
        Ti = 192.5 #
        dt_T = np.inf
        dT = 0.5

        Hi = 0.
        H0 = 20000.
        dt_H = 1/0.83/2. # dt_H <= dt
        dH = 100./2.  # 10 Oe = 1 mT
    
    plot_evolutions.has_been_called = 0
    Htype = 3
    
    print 'comp = %g, v = %g, ti = %g, tf = %g, dt = %g' % (compound, v, t[0], t[-1], dt)
    print 'T_min = %g, T_max = %g, Ti = %g, dT = %g, dt_T = %g' % (T_min, T_max, Ti, dT, dt_T)
    print 'Hi = %g, H0 = %g, dH = %g, dt_H = %g, Htype = %g' % (Hi, H0, dH, dt_H, Htype)
    print 'compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(3)' % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype)
    print '\n\n'
#    plot_G(M, Ti_G, Tf_G, dT_G, Hi_G, Hf_G, dH_G, a, b, c)
    entropy_equilibrium(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype)
#    plot_dGdM(M, Ti_G, Tf_G, dT_G, Hi_G, Hf_G, dH_G, a, b, c)
    
    plot_T(t, dt_T, T_min, T_max, Ti, dT)
    plot_H(t, dt_H, Hi, H0, dH, Htype)
#    plot_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype)
#    plot_evolutions(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype)
#    plot_M_T(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype)
#    plot_dMdT_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H)
#    plot_entropy(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype)