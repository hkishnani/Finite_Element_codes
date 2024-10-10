#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final for assignment submission

Created on Sun Aug 25 05:24:16 2024

@author: himanshu
"""

import numpy as np
from numpy import zeros
import scipy as sp
from scipy.sparse import spdiags


''' 
    ----------------  1
         \  /
          \/ 
          /\
    _____/__\______  0
       0      1
'''
#%% Assuming linear nodal basis
def phi_L_0(x:float, x_lim:tuple):
    result = (x_lim[1] - x       )/ \
             (x_lim[1] - x_lim[0])    
    if(result >= 0 and result <= 1):
        return result
    else:
        return 0


def phi_L_1(x:float, x_lim:tuple):
    result = (x        - x_lim[0])/ \
             (x_lim[1] - x_lim[0])    
    if(result >= 0 and result <= 1):
        return result
    else:
        return 0


#%% differentiation
def d_phi(f, x:float, x_lim:tuple):
    result = 0.0
    h      = 1e-6
    k      = 0.0
    l      = 0.0
    Dr     = h
    
    if (x < x_lim[1] and x >= x_lim[0]):
        k = h
    if (x > x_lim[0] and x <= x_lim[1]):
        l = h
    
    if( (k+l) > 0):
        Dr = k+l
    
    result = ( f(x+k, x_lim) - f(x-l, x_lim) ) / Dr
    
    return result


#%% integration function
def integrate_phi(f, x_lim, order):
    h = x_lim[1] - x_lim[0]
    m = 0.5 * (x_lim[1] + x_lim[0])
    
    if(order == 0):
        phi_0 = f(x_lim[0], x_lim)
        phi_M = f(m       , x_lim)
        phi_1 = f(x_lim[1], x_lim)
    
    elif(order == 1):
        phi_0 = d_phi(phi_L_0, x_lim[0], x_lim)
        phi_M = d_phi(phi_L_0, m       , x_lim)
        phi_1 = d_phi(phi_L_0, x_lim[1], x_lim)
    
    return (phi_0 + 4*phi_M + phi_1) * (h/6)


def integrate_phi_i_x_phi_j(x_lim, order):
    h = x_lim[1] - x_lim[0]
    m = 0.5 * (x_lim[0] + x_lim[1])
    phi_0 = zeros(2)
    phi_M = zeros(2)
    phi_1 = zeros(2)

    A = zeros((2, 2))

    if (order == 0):
        phi_0[0] = phi_L_0(x_lim[0], x_lim)
        phi_M[0] = phi_L_0(m       , x_lim)
        phi_1[0] = phi_L_0(x_lim[1], x_lim)

        phi_0[1] = phi_L_1(x_lim[0], x_lim)
        phi_M[1] = phi_L_1(m       , x_lim)
        phi_1[1] = phi_L_1(x_lim[1], x_lim)


    if (order == 1):
        phi_0[0] = d_phi(phi_L_0, x_lim[0], x_lim)
        phi_M[0] = d_phi(phi_L_0, m       , x_lim)
        phi_1[0] = d_phi(phi_L_0, x_lim[1], x_lim)

        phi_0[1] = d_phi(phi_L_1, x_lim[0], x_lim)
        phi_M[1] = d_phi(phi_L_1, m       , x_lim)
        phi_1[1] = d_phi(phi_L_1, x_lim[1], x_lim)

    for i in range(2):
        for j in range(2):
            A[i,j] = (     phi_0[i] * phi_0[j]
                     + 4 * phi_M[i] * phi_M[j]
                     +     phi_1[i] * phi_1[j]
                     ) * (h/6)

    return A


def integrate_f_phi_elem(f, x_lim):
    m = 0.5 * (x_lim[0] + x_lim[1])
    vec = zeros(2)
    
    h = x_lim[1] - x_lim[0]
    
    phi_0 = zeros(2)
    phi_M = zeros(2)
    phi_1 = zeros(2)
    
    phi_0[0] = phi_L_0(x_lim[0], x_lim)
    phi_M[0] = phi_L_0(m       , x_lim)
    phi_1[0] = phi_L_0(x_lim[1], x_lim)

    phi_0[1] = phi_L_1(x_lim[0], x_lim)
    phi_M[1] = phi_L_1(m       , x_lim)
    phi_1[1] = phi_L_1(x_lim[1], x_lim)
    
    for i in range(2):
        vec[i] = (     phi_0[i] * f(x_lim[0])
                  +4 * phi_M[i] * f(m)
                  +    phi_1[i] * f(x_lim[1])
                 ) * (h/6)

    return vec