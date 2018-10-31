#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:20:14 2017

@author: rui
"""

compound = 1

# Gd5Si2Ge2
if compound == 1:
    struct1 = 'M'
    struct2 = 'O(I)'
    J = 7 / 2.
    gJ = 2.
    Tc1 = 251.
    Tc2 = 308.
    DeltaF = 0.36
    ThetaD1 = 250.
    ThetaD2 = 278.
    N = 36.
    Nm = 20.


if compound == 2:
    struct1 = 'M'
    struct2 = 'O(I)'
    J = 7 / 2.
    gJ = 2.
    Tc1 = 450.
    Tc2 = 308.
    DeltaF = 0.36
    ThetaD1 = 250.
    ThetaD2 = 278.
    N = 36.
    Nm = 20.



if Tc2 < Tc1:
    temp = Tc1
    Tc1 = Tc2
    Tc2 = temp
    
    temp = ThetaD1
    ThetaD1 = ThetaD2
    ThetaD2 = temp
    
    temp = struct1
    struct1 = struct2
    struct2 = temp

if __name__ == "__main__":
    print struct1, struct2
    print J, gJ
    print 'Tc--', Tc1, Tc2
    print 'ThetaD--', ThetaD1, ThetaD2
    print DeltaF
    print N, Nm
