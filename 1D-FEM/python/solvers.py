#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 07:51:18 2024

@author: himanshu
"""

import numpy as np

def forward_substitution(A, b):

    n = A.shape[0]
    m = b.shape[0]

    if m!=n:
        print("Dim not matching")
        exit

    for i in range(0, n):
        for j in range(0, i):
            b[i] = b[i] - A[i,j] * b[j]

        b[i] = b[i]/A[i,i]
    return b

def backward_substitution(A, b):

    n = A.shape[0]
    m = b.shape[0]

    if m!=n:
        print("Dim not matching")
        exit


    for i in range(n-1,-1,-1):
        for j in range(n-1, i, -1):
            b[i] = b[i]-A[i,j]*b[j]

        b[i] = b[i]/A[i,i]

    return b

def cholesky_decomposition(A):
    n = A.shape[0]

    for i in range(n):
        if i != 0:
            for k in range(i):
                A[i,i] = A[i,i] - A[k, i]**2

        if A[i,i] <= 0:
            print("A is not Positive definite")
            quit()
            
        A[i,i] = A[i,i]**0.5
        
        if i!=n-1:
            for j in range(i+1, n):
                if i!=0:
                    for k in range(i):
                        A[i,j] = A[i,j] - A[k,i]*A[k,j]
                A[i,j] = A[i,j]/A[i,i]

    return np.triu(A)