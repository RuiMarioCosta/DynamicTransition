# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 09:49:52 2017

@author: Rui
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, signal, integrate
from scipy.signal import argrelmin
from scipy.integrate import odeint
import multiprocessing
import time
import pickle
import os
import glob


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


def mag_indices(M, T, H, a, b, c):
    return argrelmin(G(M, T, H, a, b, c))[0]


def M_T(M, T, H, a, b, c):
    mags_left = np.zeros_like(T) * np.nan
    mags_mid = np.zeros_like(T) * np.nan
    mags_right = np.zeros_like(T) * np.nan
    for i in range(len(T)):
        mags = M[mag_indices(M, T[i], H, a, b, c)]
        if len(mags) == 3:
            # print 3333, T[i], H, mags
            mags_left[i] = mags[0]
            mags_mid[i] = mags[1]
            mags_right[i] = mags[2]
        elif len(mags) == 2:
            temp = np.abs(mags) > 50.
            # print 2222, T[i], H, mags, temp
            if temp[0] and temp[0]:
                mags_left[i] = mags[0]
                mags_right[i] = mags[1]
            else:
                if H < 0.:
                    mags_left[i] = mags[0]
                    mags_mid[i] = mags[1]
                else:
                    mags_mid[i] = mags[0]
                    mags_right[i] = mags[1]
        else:
            # print 1111, T[i], H, mags
            mags_mid[i] = mags
    return mags_left, mags_mid, mags_right


def M_H(M, T, H, a, b, c):
    mags_left = np.zeros_like(H) * np.nan
    mags_mid = np.zeros_like(H) * np.nan
    mags_right = np.zeros_like(H) * np.nan
    for i in range(len(H)):
        mags = M[mag_indices(M, T, H[i], a, b, c)]
        if len(mags) == 3:
            #            print 3333, T, H[i], mags
            mags_left[i] = mags[0]
            mags_mid[i] = mags[1]
            mags_right[i] = mags[2]
        elif len(mags) == 2:
            temp = np.abs(mags) > 50.
#            print 2222, T, H[i], mags, temp
            if temp[0] and temp[0]:
                mags_left[i] = mags[0]
                mags_right[i] = mags[1]
            else:
                if H[i] < 0.:
                    mags_left[i] = mags[0]
                    mags_mid[i] = mags[1]
                else:
                    mags_mid[i] = mags[0]
                    mags_right[i] = mags[1]
        else:
            #            print 1111, T, H[i], mags
            if np.abs(
                    mags) < 50.:  # if magnetization difference is less than 10 emu
                mags_mid[i] = mags
            # if magnetization difference is less than 10 emu
            elif np.abs(mags - mags_left[i - 1]) < 10.:
                mags_left[i] = mags
            else:
                mags_right[i] = mags
#        print '-------'
    return mags_left, mags_mid, mags_right


def M_TH(M, T, H, a, b, c):
    # variable to store values of magnetization in the plane T-H
    M_2D = np.zeros((len(T), len(H)))
    for j in xrange(len(H)):
        for i in xrange(len(T)):
            if H[j] == 0.:
                M_2D[i, j] = -M[np.argmin(G(M, T[i], H[j], a, b, c))]
            else:
                M_2D[i, j] = M[np.argmin(G(M, T[i], H[j], a, b, c))]

    return M_2D

def S_TH(M, T, H, a, b, c):
    G_min = np.zeros((len(T), len(H)))
    print G_min.shape
    for i in xrange(len(T)):
        for j in xrange(len(H)):
            G_min[i,j] = np.min(G(M, T[i], H[j], a, b, c))
    dGdT, dGdH = np.gradient(G_min, T[1]-T[0], H[1]-H[0])
    return -dGdT, dGdH

def magnetization(M, T, H, task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            # Poison pill/sentinel means shutdown
            break
        i, j = task
        t = T[i]
        h = H[j]
        mag = M[np.argmin(G(M, t, h))]  # compute magnetization
        # send indices of position and magnetization
        result_queue.put((i, j, mag))
        task_queue.task_done()  # signal task completion
    return


def M_TH_multi(M, T, H):
    # Establish communication queues
    task_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.Queue()

    N_proc = 4  # number of processes
    for proc in xrange(N_proc):
        proc = multiprocessing.Process(target=magnetization,
                                       args=(M, T, H, task_queue, result_queue))
        proc.start()

    tasks = [(i, j) for i in xrange(len(T)) for j in xrange(len(H))]

    # enqueue tasks
    for task in tasks:
        task_queue.put(task)  # put indices of temperature and magnetic field

    # add a poison pill for each consumer
    for i in xrange(N_proc):
        task_queue.put(None)

    # wait for all the tasks to finish
    task_queue.join()

    # magnetization array
    M_2D = np.zeros((len(T), len(H)))

    # insert results into magnetization array
    for i in xrange(len(tasks)):
        i, j, mag = result_queue.get()
        if H[j] == 0.:
            M_2D[i, j] = -mag
        else:
            M_2D[i, j] = mag

    return M_2D


def dMdT(M, T, H):
    mag = M_TH_multi(M, T, H)

    dT = T[1] - T[0]
    dH = H[1] - H[0]
    dMdT, dMdH = np.gradient(mag, dT, dH)
    return dMdT


def dMdT_th(M, T, H):
    mag = M_TH_multi(M, T, H)
    H, T = np.meshgrid(H, T)
    return -A * mag / (A * (T - Tc) + 3. * B * mag**2. + 5. * C * mag**4.)


def phase_diagram(M, T, H, a, b, c):
    diagram = np.zeros((len(T), len(H)))
    for i in range(len(T)):
        for j in range(len(H)):
            m_ind = mag_indices(M, T[i], H[j], a, b, c)
            if len(m_ind) == 3:
                diagram[i, j] = 3
            elif len(m_ind) == 2:
                diagram[i, j] = 2
            else:
                diagram[i, j] = 1
    return diagram


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
        If Htype = 0, maximum applied magnetic field.
        If Htype = 1, final magnetic field.
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
            if t[i] > n * dt:
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
    return B


def magnetization_relaxation(M0, M_avg, t, v):
    return M_avg + (M0 - M_avg) * np.exp(-v * t)


def evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
              Hi, H0, dH, dt_H, Htype, hysteresis):

    # file name of magnetization
    file_name = ('evolution, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
    'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(%g).pickle')\
    % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
       Hi, H0, dH, dt_H, Htype, hysteresis)

    # directory's path and file name being executed
    dirname, main_filename = os.path.split(os.path.abspath(__file__))

    # if file does not exists in data folder, do calculations
    if find(file_name, dirname + '\data') == None:
        print '--File not found - evolution'
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
        if hysteresis:
            for i in xrange(0, len(t) - 1):
                rel_mins = signal.argrelmin(G(M, T[i], H[i], a, b, c))[0]
                if len(rel_mins) == 1:
                    M_min = M[np.argmin(G(M, T[i], H[i], a, b, c))]
                else:
                    M0 = M[rel_mins]
                    M_min = M0[0]
                    for m in M0[1:]:
                        if np.abs(m - mag_evolution[i]) < np.abs(M_min - mag_evolution[i]):
                            M_min = m

                mag_evolution[i + 1] = magnetization_relaxation(mag_evolution[i],
                                                                M_min, dt, v)
        else:
            for i in xrange(0, len(t) - 1):
                M_min = M[np.argmin(G(M, T[i], H[i], a, b, c))]
                mag_evolution[i + 1] = magnetization_relaxation(mag_evolution[i],
                                                                M_min, dt, v)
#            print i, t[i], T[i], H[i], M_min, rel_mins

        pickle_save(mag_evolution, 'data/' + file_name)  # save data to a file

    # if file exists, load it
    else:
        mag_evolution = pickle_load('data/' + file_name)
        print '--File exists'

    return mag_evolution


def M_Tt(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
         Hi, H0, dH, dt_H, Htype, hysteresis):

    file_name = ('magnetization, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
    'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(%g).pickle')\
    % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
       Hi, H0, dH, dt_H, Htype, hysteresis)
    
    dirname, main_filename = os.path.split(os.path.abspath(__file__))

    # if file does not exists in data folder
    if find(file_name, dirname + '\data') == None:
        print '--File not found - M_Tt'
        N_pts = int((T_max - T_min) / dT)  # number of temperatures
        mag = np.zeros((N_pts, len(t)))  # magnetization variable
        # calculate the magnetization for the various temperatures
        for i in range(N_pts):
            print '----', i, N_pts, Ti
            mag[i] = evolution(compound, M, dm, a, b, c, v, t, dt, T_min,
                               T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype,
                               hysteresis)
            Ti += dT  # temperature increase

        # save to a file
        pickle_save(mag, 'data/' + file_name)

    # if file exists
    else:
        mag = pickle_load('data/' + file_name)

    return mag


def dMdT_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                   dt_T, Hi, H0, dH, dt_H, Htype, hysteresis):
    
    file_name = ('dMdT_evolution, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
    'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(%g).pickle')\
    % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
       Hi, H0, dH, dt_H, Htype, hysteresis)
    
    dirname, main_filename = os.path.split(os.path.abspath(__file__))

    # if file does not exists in data folder
    if find(file_name, dirname + '\data') == None:
        print '--File not found - dMdT_evolution'
        mag = M_Tt(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                   dt_T, Hi, H0, dH, dt_H, Htype, hysteresis)
        dMdT, dMdt = np.gradient(mag, dT, dt)  # compute gradient

        pickle_save(dMdT, 'data/' + file_name)  # save to a file
    # if file exists
    else:
        dMdT = pickle_load('data/' + file_name)  # load magnetization

    return dMdT  # return temperature derivative


def entropy(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
            Hi, H0, dH, dt_H, Htype, hysteresis):
    
    file_name = ('entropy, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
    'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(%g).pickle')\
    % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
       Hi, H0, dH, dt_H, Htype, hysteresis)
    
    dirname, main_filename = os.path.split(os.path.abspath(__file__))

    # if file does not exists in data folder
    if find(file_name, dirname + '\data') == None:
        print '--File not found - entropy'
        dMdT = dMdT_evolution(compound, M, dm, a, b, c, v, t, dt, T_min,
                              T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype,
                              hysteresis)

        # H needs to be partitioned otherwise there are multiple magnetizations
        # for the same magnetic field
        dH_interval = int(dt_H / dt)
        H = B_applied(t, dt_H, Hi, H0, dH, Htype)[1::dH_interval]  # magnetic field
        dMdT = dMdT[:, 1::dH_interval]
        max_indices = signal.argrelmax(H)[0]
        min_indices = signal.argrelmin(H)[0]
        N_pts = int((T_max - T_min) / dT)  # number of temperatures

#        print dMdT, dMdT.shape, H.shape, max_indices, min_indices
        # the integration is done in pieces, i.e., the first goes from H=0 to H=H0,
        # resulting in the entropy change, S(H0) - S(0), the next one goes from
        # H=H0 to H=-H0 resulting in S(-H0) - S(H0), and so on until there are no
        # more.
        N_pieces = len(max_indices) + len(min_indices) + 1
        ent = np.zeros((N_pts, N_pieces))

        # create variable that keeps limits of integration
        lim_int = np.zeros(N_pieces + 1, int)
        lim_int[-1] = -1
        for i in xrange(len(max_indices)):
            lim_int[2 * i + 1] = max_indices[i]
        for i in xrange(len(min_indices)):
            lim_int[2 * i + 2] = min_indices[i]

#        print 'dMdT', dMdT[0, lim_int[0]:lim_int[1]]
#        print ent.shape, lim_int
        for i in xrange(N_pts):
            for j in xrange(N_pieces):
                #            print j, 'H', H[lim_int[j]:lim_int[j]+4], H[lim_int[j+1]]
                ent[i, j] = integrate.simps(dMdT[i, lim_int[j]:lim_int[j + 1]],
                                            H[lim_int[j]:lim_int[j + 1]])

            print '----', i, Ti, N_pts
            Ti += dT

        pickle_save(ent, 'data/' + file_name)  # save to a file

    # if file exists
    else:
        ent = pickle_load('data/' + file_name)  # load magnetization

    return ent  # integrate to obtain entropy


def pickle_save(obj, file_name):
    pickle_out = open(file_name, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def pickle_load(file_name):
    return pickle.load(open(file_name, 'rb'))


def txt_save():
    # directory's path and file name being executed
    dirname, main_filename = os.path.split(os.path.abspath(__file__))
    print dirname+'\data'
    for filename in os.listdir('data'):
#        print filename
        filedata = pickle_load('data\\'+filename)
#        print filedata.shape
        filename = filename[:-7] + '.txt' # remove .pickle and add .txt extension
#        print filename
        np.savetxt('data-txt\\' + filename, filedata.T, delimiter=' ')
        
        
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def time_dif(time_var):
    print time.time() - time_var
    time_var = time.time()
    return


#==============================================================================
def plot_ABC(T, a, b, c):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax3.plot(T, A(T, *a), '--o', label='A')
    ax2.plot(T, B(T, *b), '--o', label='B')
    ax1.plot(T, C(T, *c), '--o', label='C')
    ax3.set_ylabel('A(Oe g/emu)')
    ax2.set_ylabel('B(Oe g^3/emu^3)')
    ax1.set_ylabel('C(Oe g^5/emu^5)')
    ax3.set_xlabel('T(K)')
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()
    print 'plot_ABC'


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


def plot_G_T(M, Ti, Tf, dT, Hi, Hf, dH, a, b, c):
    for i in range(Ti, Tf, dT):
        plt.figure('plot_G, T=%g' % T[i])
        for j in range(Hi, Hf, dH):
            plt.plot(M, G(M, T[i], H[j], a, b, c), label='T=%g' % H[j])
            plt.legend(loc=0, fontsize='xx-small')
        plt.grid()
        plt.ylabel('G')
        plt.xlabel('M (emu/g)')
    plt.show()
    print 'plot_G_T'
    

def plot_dGdM(M, Ti, Tf, dT, Hi, Hf, dH, a, b, c):
    for i in range(Hi, Hf, dH):
        plt.figure('plot_G, H=%g' % H[i])
        for j in range(Ti, Tf, dT):
            plt.plot(M, dGdM(M, T[j], H[i], a, b, c), label='T=%g' % T[j])
            plt.legend(loc=0, fontsize='xx-small')
        plt.grid()
        plt.ylabel('G(emu G/g, erg/g)')
        plt.xlabel('M (emu/g)')
    plt.show()
    print 'plot_G'


def plot_M_T(M, T, H, a, b, c):
    plt.figure('plot_M_T')
    di = int(len(H) / 3)
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(H) + di * 2 / 3))
    for i in range(0, len(H), di):
        Mags = M_T(M, T, H[i], a, b, c)
        plt.plot(T, Mags[0], marker=None, c=colors[i],
                 label='left, %g Oe' % H[i], markersize=1)
        plt.plot(T, Mags[1], marker=None, c=colors[i + di / 3],
                 label='mid, %g Oe' % H[i], markersize=1)
        plt.plot(T, Mags[2], marker=None, c=colors[i + di * 2 / 3],
                 label='right, %g Oe' % H[i], markersize=1)
        plt.legend(loc=0, fontsize='xx-small')
        plt.ylabel('M(emu/g)')
        plt.xlabel('T(K)')
        plt.grid()
    plt.show()
    print 'plot_M_T'


def plot_M_H(M, T, H, a, b, c):
    plt.figure('plot_M_H')
    di = 1 #int(len(T) / 6)
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(T) + di * 2 / 3))
    for i in range(0, len(T), di):
        Mags = M_H(M, T[i], H, a, b, c)
        plt.plot(H, Mags[0], marker=None, c=colors[i],
                 label='left, %g K' % T[i], markersize=1)  # b
        plt.plot(H, Mags[1], marker=None, c=colors[i + di / 3],
                 label='mid, %g K' % T[i], markersize=1)  # y
        plt.plot(H, Mags[2], marker=None, c=colors[i + di * 2 / 3],
                 label='right, %g K' % T[i], markersize=1)  # g
        plt.legend(loc=0, fontsize='xx-small')
        plt.ylabel('M(emu/g)')
        plt.xlabel('H(Oe)')
        plt.grid()
        
        if T[i] == 195.:
            print 'saving M-H for T=195 K'
            file_data = np.vstack((H, Mags)).T # save data to .txt
            np.savetxt('data-txt\\Landau-M-H-195K.txt', file_data, delimiter=' ')
    plt.show()
    print 'plot_M_H'

#    plt.figure('Arrot')
#    for i in range(0, len(T), int(len(T) / 6)):
##        plt.figure('plot_M_H, M(T=%g,H)' % T[i])
#        plt.plot(Mags[0]**2., H/Mags[0], 'o', c=colors[i], label='left, %g K' % T[i], markersize=1)
#        plt.plot(Mags[1]**2., H/Mags[1], 'o', c=colors[i], label='mid, %g K' % T[i], markersize=1)
#        plt.plot(Mags[2]**2., H/Mags[2], 'o', c=colors[i], label='right, %g K' % T[i], markersize=1)
#        plt.legend(loc=0, fontsize='xx-small')
#        plt.ylabel('H/M(Oe g/emu)')
#        plt.xlabel('M^2(emu^2/g^2)')
#        plt.grid()
#    plt.show()
#    print 'Arrot'


def plot_M_TH(M, T, H, a, b, c):
    plt.figure('plot_M_TH')
#    Mag = M_TH_multi(M, T, H)
    Mag = M_TH(M, T, H, a, b, c)
    im = plt.imshow(Mag,
                    extent=[H[0], H[-1], T[0], T[-1]],
                    aspect='auto',
                    origin='lower')
    plt.colorbar(im)
    plt.xlabel('H(Oe)')
    plt.ylabel('T(K)')


#    from mpl_toolkits.mplot3d import Axes3D
#    from matplotlib import cm
#    from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
#    fig = plt.figure('M_TH 3D')
#    ax = fig.gca(projection='3d')
#
#    H, T = np.meshgrid(H, T)
#
#    # Plot the surface.
#    surf = ax.plot_surface(T, H, Mag, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
#
#    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    print 'plot_M_TH'


def plot_dMdT(M, T, H):
    plt.figure('plot_dMdT_num')
    dMdT_num = dMdT(M, T, H)
    im = plt.imshow(dMdT_num,
                    extent=[H[0], H[-1], T[0], T[-1]],
                    aspect='auto',
                    origin='lower')
    plt.colorbar(im)
    plt.xlabel('H(Oe)')
    plt.ylabel('T(K)')

    plt.figure('plot_dMdT_the')
    dMdT_the = dMdT_th(M, T, H)
    im = plt.imshow(dMdT_the,
                    extent=[H[0], H[-1], T[0], T[-1]],
                    aspect='auto',
                    origin='lower')
    plt.colorbar(im)
    plt.xlabel('H(Oe)')
    plt.ylabel('T(K)')

    plt.show()
    print 'plot_dMdT'
    
def plot_S_H(M, T, H, a, b, c):
    plt.figure('plot_S_H')
    S, dGdH = S_TH(M, T, H, a, b, c)
    print 'S', S.shape
    for i in range(0, len(T), 5):
        plt.plot(H, S[i], marker=None, label='left, %g K' % T[i], markersize=1)
        plt.legend(loc=0, fontsize='xx-small')
        plt.ylabel('S')
        plt.xlabel('H(Oe)')
        plt.grid()
    plt.show()
    print 'plot_M_H'


def plot_phase_diagram(M, T, H, a, b, c):
    plt.figure('Phase Diagram')
    im = plt.imshow(phase_diagram(M, T, H, a, b, c),
                    extent=[H[0], H[-1], T[0], T[-1]],
                    aspect='auto',
                    origin='lower')
#    im = plt.imshow(phase_diagram(M, T, H, a, b, c).T,
#                    extent=[T[0], T[-1], H[0], H[-1]],
#                    aspect='auto',
#                    origin='lower')
    plt.colorbar(im)
    plt.xlabel('H(Oe)')
    plt.ylabel('T(K)')
    plt.show()
    print 'plot_phase_diagram'

    plt.savefig('results/phase-diagram.png', bbox_inches='tight', dpi=600)


def plot_T(t, dt, T_min, T_max, Ti, dT):
    print 'plot_T'
    plt.figure('Temperature')
    plt.plot(t, T_applied(t, dt, T_min, T_max, Ti, dT))
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


def plot_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                   dt_T, Hi, H0, dH, dt_H, Htype, hysteresis,
                   fig=None, axes=None):
    Mag = evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                    dt_T, Hi, H0, dH, dt_H, Htype, hysteresis)

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
                  'Htype = %g, hysteresis = %g') %
            (compound, v, dT/dt_T, dH/dt_H, Htype, hysteresis))
    t_shift = H0/dH*dt_H
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
            m_18mTs_8000Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-8000Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_9000Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-9000Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_7200Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-7200Oe-Mt.txt',
                                       delimiter='\t')
            m_18mTs_9800Oe_Mt = np.loadtxt('LaFe11.6Si1.4-Lovell-18mTs-9800Oe-Mt.txt',
                                       delimiter='\t')
            ax6.plot(m_18mTs_7200Oe_Mt[:, 0], m_18mTs_7200Oe_Mt[:, 1], 'ok',
                     label='Lovell 18 mT/s, 072 T', ms=1)
            ax6.plot(m_18mTs_8000Oe_Mt[:, 0], m_18mTs_8000Oe_Mt[:, 1], 'or',
                     label='Lovell 18 mT/s, 0.8 T', ms=1)
            ax6.plot(m_18mTs_9000Oe_Mt[:, 0], m_18mTs_9000Oe_Mt[:, 1], 'og',
                     label='Lovell 18 mT/s, 0.9 T', ms=1)
            ax6.plot(m_18mTs_9800Oe_Mt[:, 0], m_18mTs_9800Oe_Mt[:, 1], 'ob',
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
    ax6.set_xlim(-50, 100)
    ax6.set_ylim(0,120)
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
    print 'plot_evolution'
    
    if 1 == 1: # save
        file_name = ('evolution, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
                             'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(%g).png')\
                             % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
                                Hi, H0, dH, dt_H, Htype, hysteresis)
        
        # save data to .txt
        file_data = np.vstack((t, T, H, Mag)).T # save data to .txt
        np.savetxt('data-txt\\' + file_name[:-3]+'txt', file_data, delimiter=' ')
        
        # save figure
        if plot_evolutions.has_been_called == -1 or plot_evolutions.has_been_called == 0:  # if last called
            if plot_evolutions.has_been_called == -1: # name: evolutions
                file_name = ('evolutions, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), '
                             'T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g, %g), Hyst(%g).png')\
                             % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T,
                                Hi, H0, dH, dt_H, Htype, hysteresis)
    
            # directory's path and file name being executed
            dirname, main_filename = os.path.split(os.path.abspath(__file__))
            # if file does not exists in data folder, do calculations
            if find(file_name, dirname + '\\results') == None:
                print '...saving - plot_evolution'
                fig.savefig('results/' + file_name, bbox_inches='tight', dpi=300)
            else:
                print file_name, '--File already saved.'
        
        
            

def plot_evolutions(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                    dt_T, Hi, H0, dH, dt_H, Htype, hysteresis):
    plot_evolutions.has_been_called = 0
    N_pts = int((T_max - T_min) / dT)
    Ti = T_min
    n = 1
    list_range = [i for i in xrange(0, N_pts, n)]
    fig, axes = plt.subplots(nrows = 2, ncols=3, num='plot_evolutions')
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    for i in list_range:
        Ti += dT * n
        plot_evolutions.has_been_called += 1
        if i == list_range[-1]:
            plot_evolutions.has_been_called = -1
        print '----', i, N_pts, Ti
        plot_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti,
                       dT, dt_T, Hi, H0, dH, dt_H, Htype, hysteresis, fig, axes)
    print 'plot_evolutions'


def plot_M_Tt(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
              Hi, H0, dH, dt_H, Htype, hysteresis):
    MT_evol = M_Tt(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                   dt_T, Hi, H0, dH, dt_H, Htype, hysteresis)

    plt.figure('plot_M_Tt_evolution')
    im = plt.imshow(MT_evol,
                    extent=[t[0], t[-1], T[0], T[-1]],
                    aspect='auto',
                    origin='lower')
    plt.colorbar(im)
    plt.xlabel('t')
    plt.ylabel('T(K)')
    plt.title('compound=%g, v = %g, dT/dt = %g, dH/dt = %g' %
              (compound, v, dT / dt_T, dH / dt_H))

    plt.show()
    print 'plot_MT_evolution'


def plot_dMdT_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti,
                        dT, dt_T, Hi, H0, dH, dt_H):
    dMdT_evol = dMdT_evolution(compound, M, dm, a, b, c, v, t, dt, T_min,
                               T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype,
                               hysteresis)

    plt.figure('plot_dMdT_evolution')
    im = plt.imshow(dMdT_evol,
                    extent=[t[0], t[-1], T[0], T[-1]],
                    aspect='auto',
                    origin='lower')
    plt.colorbar(im)
    plt.xlabel('t')
    plt.ylabel('T(K)')
    plt.title('compound=%g, v = %g, dT/dt = %g, dH/dt = %g' %
              (compound, v, dT / dt_T, dH / dt_H))

    plt.show()
    print 'plot_dMdT_evolution'


def plot_entropy(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                 dt_T, Hi, H0, dH, dt_H):
    ent = entropy(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
                  dt_T, Hi, H0, dH, dt_H, Htype, hysteresis)
    T = np.arange(T_min, T_max, dT)

    plt.figure('plot_entropy, compound=%g, v = %g, dTdt = %g, dHdt = %g, dT = %g, dH = %g, dt = %g' % (
        compound, v, dT / dt_T, dH / dt_H, dT, dH, dt), figsize=(15.0, 7.0))
    for i in xrange(len(ent[0])):
        plt.plot(T, ent[:, i], label='%d' % i, marker='o', ms=3.)

    plt.legend(loc=0)
    plt.xlabel('T(K)')
    plt.ylabel('S')
    plt.title('compound = %g, v = %g $s^{-1}$, H0 = %g Oe \n dT = %g K, dH = %g Oe, dt = %g s, dT/dt = %g K/s, dH/dt = %g Oe/s' %
              (compound, v, H0, dT, dH, dt, dT / dt_T, dH / dt_H))

    plt.show()
    print 'plot_entropy'

    file_name = 'entropy, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g)' % (
        compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H)
    plt.savefig('results/' + file_name + '.png', bbox_inches='tight', dpi=600)

#    T = np.reshape(T, (len(T),1))
#    ent = np.append(T, ent, axis=1)
#    file_name = 'data/entropy, compound(%g), M(%g,%g), v(%g), t(%g, %g, %g), T(%g, %g, %g, %g, %g), H(%g, %g, %g, %g).pickle' % (compound, M[-1], dm, v, t[0], t[-1], dt, T_min, T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H)
#    np.savetxt(file_name, ent, delimiter='\t', header='Temperature\tHalf up\tFull down\tFull up\tHalf down')


if __name__ == '__main__':
    compound = 4
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

        dm = 1e0
        dt = 1e0
        dh = 1e3
        M = np.arange(-200, 200 + dm, dm)
        T = np.arange(190, 200. + dt, dt)
        H = np.arange(0., 30000. + dh, dh)

        Ti_G = 0  # starting index
        Tf_G = len(T) - int(0 / dt)  # ending index
        dT_G = int(1 / dt)  # index step
        Hi_G = 0  # starting magnetic field
        Hf_G = len(H)  # ending magnetic field
        dH_G = int(10000 / dh)  # index step

        v = 0.2  # attempt frequency

        dt = 1. # dt <= dt_H
        t = np.arange(0, 601 + dt, dt)

        T_min = 192.5
        T_max = 196.
        Ti = 194.5
        dt_T = np.inf
        dT = 0.5

        Hi = 0.
        H0 = 15000.
        dt_H = 1. # dt <= dt_H
        dH = 180. # 10 Oe = 1 mT

    elif compound == 4:  # Amaral, Tc = 195 K, Ms = 100 emu/g, FOPT
        a = [-4500., 25, 0, 0, 0, 0]
        b = [-0.18, 0, 0, 0, 0, 0]
        c = [2.33e-5, 0, 0, 0, 0, 0]

        dm = 1e-1
        dt = 1e0
        dh = 1e2
        M = np.arange(-150, 150 + dm, dm)
        T = np.arange(185, 220. + dt, dt)
        H = np.arange(0., 20000. + dh, dh)

        Ti_G = 0  # starting index
        Tf_G = len(T) - int(0 / dt)  # ending index
        dT_G = int(10 / dt)  # index step
        Hi_G = 0  # starting magnetic field
        Hf_G = len(H)  # ending magnetic field
        dH_G = int(10000 / dh)  # index step

        v = 1.  # attempt frequency

        dt = 0.1
        t = np.arange(0, 90. + 2*dt, dt)

        T_min = 186.
        T_max = 212.
        Ti = 195.
        dt_T = np.inf
        dT = 1.

        Hi = 0.
        H0 = 15000.
        dt_H = 0.1 # dt_H <= dt
        dH = 100.  # 10 Oe = 1 mT
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
        t = np.arange(0, 600. * 10. / 1.8 + dt, dt)

        T_min = 170.
        T_max = 230.
        Ti = 170.
        dt_T = np.inf
        dT = 0.5

        Hi = 0.
        H0 = 15000.
        dt_H = 1. / 1.
        dH = 100. * 1.8 / 10.  # 10 Oe = 1 mT
    elif compound == 6: # LaFe11.6Si1.4, FOPT, Tc = 191 K, T = 194.5 K, Ms = 120 g/emu
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

        T_min = 192.5
        T_max = 196.
        Ti = 192.5 #
        dt_T = np.inf
        dT = 0.5

        Hi = 0.
        H0 = 20000.
        dt_H = 1/0.83/2. # dt_H <= dt
        dH = 100./2.  # 10 Oe = 1 mT
        

    plot_evolutions.has_been_called = 0
    Htype = 0
    hysteresis = 0
#    plot_ABC(T, a,b ,c)
#    plot_G(M, Ti_G, Tf_G, dT_G, Hi_G, Hf_G, dH_G, a, b, c)
#    plot_G_T(M, Ti_G, Tf_G, dT_G, Hi_G, Hf_G, dH_G, a, b, c)

#    plot_M_T(M, T, H, a, b, c)
    plot_M_H(M, T, H, a, b, c)
#    plot_S_H(M, T, H, a, b, c)

#    plot_M_TH(M, T, H, a, b, c)
#    plot_dMdT(M, T, H)

#    plot_phase_diagram(M, T, H, a, b, c)

#    plot_T(t, dt_T, T_min, T_max, Ti, dT)
#    plot_H(t, dt_H, Hi, H0, dH, Htype)

#    plot_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
#                   dt_T, Hi, H0, dH, dt_H, Htype, hysteresis)
#    plot_evolutions(compound, M, dm, a, b, c, v, t, dt, T_min,
#                    T_max, Ti, dT, dt_T, Hi, H0, dH, dt_H, Htype, hysteresis)
#
#    plot_M_Tt(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT, dt_T,
#              Hi, H0, dH, dt_H, Htype, hysteresis)
#    plot_dMdT_evolution(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti,
#                        dT, dt_T, Hi, H0, dH, dt_H)

#    plot_entropy(compound, M, dm, a, b, c, v, t, dt, T_min, T_max, Ti, dT,
#                 dt_T, Hi, H0, dH, dt_H)

#    txt_save()