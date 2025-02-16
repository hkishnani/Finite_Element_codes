# Condition number limit
M        = [20  , 40   , 80   , 160  , 320  , 320  ]
cond_lim = [5   , 40   , 200  , 15   , 200  , 3e6  ]
sparsity = [0   , 0.75 , 0.9  , 0.9  , 0.9  , 0.9  ]
spd      = [True, False, True , True , True , True ]
diag_mat = [True, False, False, False, True, False]
n_cases  = len(M)




import numpy as np
import scipy as sp
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
from numpy.random import random
from numpy import matrix, array, float64, zeros, eye, savetxt, diag
from numpy.linalg import norm, cond, eigvals, matrix_rank, inv
from scipy import sparse
from scipy.sparse import linalg, csc_matrix, rand
from scipy.sparse import random as sp_random
from scipy.linalg import svdvals, svd
from scipy.sparse._csc import csc_matrix as csc_type
from random import randint



def is_pd(E:matrix):
    return np.all(np.linalg.eigvals(E) > 0)

def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def is_spd(E:matrix):
    return is_symmetric(E) and is_pd(E)

def save_spd(A:matrix, str):
    savetxt( str, A )

def is_fullrank(E:matrix):
    return E.shape[0] == matrix_rank(E)

def make_dense_spd(M):
    return make_spd_matrix(M)

def make_sparse_spd(M:int, alpha:float, 
                    l_val:float, u_val:float)->matrix:
    E = make_sparse_spd_matrix(M, alpha=alpha, smallest_coef=l_val, largest_coef=u_val)
    return E

# ------------Generate a tridiagonal matrix-------------
def tridiag(A:matrix):
    # A is scipy sparse format
    k = np.arange(-1,2)
    E = A
    a = np.diag(E, k[0])
    b = np.diag(E, k[1])
    c = np.diag(E, k[2])
    return np.diag(a, k[0]) + np.diag(b, k[1]) + np.diag(c, k[2])
# ------------Generate a tridiagonal matrix-------------


# ------Generate a SPD matrix with reduced condition number--------
def reduce_cond_num(A:matrix, cond_num:int, M:int):
    i = 0
    while(cond(A) > cond_num and i < M-2):
        U, S, Vh = svd(A)
        i+=1
        S[-i] = S[-i-1]
        S = diag(S)
        A = np.dot(U, np.dot(S, Vh))

    return A

    # Using SVD to make well conditioned symmetric positive definite matrices
# ------Generate a SPD matrix with reduced condition number--------


# ---------------Tridiagonal SPD Matrix------------------
def spd_tridiag(M:int, cond_lim:int):
    A = random((M,M))
    i = 0
    while (cond(A) > cond_lim or i == 0):
        i+=1
        b = array([randint(1, 9) for i in range(M-1)])
        a = np.append(0,b) + np.append(b, 0) + array([randint(1, 9) for i in range(M)])
        A = np.diag(b, 1) + np.diag(b, -1) + np.diag(a)
    
    if is_spd(A):
        return matrix(A)
    else:
        print("Error")
# ---------------Tridiagonal SPD Matrix------------------



# ---------------Pentadiagonal SPD Matrix------------------
def spd_pentadiag(M:int, cond_lim:int):
    A = random((M,M))
    i = 0

    for i in range(1000):
        c = array([randint(1, 9) for i in range(M-10)])
        b = array([randint(1, 9) for i in range(M-1)])
        a = np.append(zeros(10), c) + np.append(c, zeros(10)) + np.append(0,b) + np.append(b, 0) + array([randint(1, 9) for i in range(M)])
        A = np.diag(c, 10) + np.diag(c, -10) + np.diag(b, 1) + np.diag(b, -1) + np.diag(a)
        reduce_cond_num(A, cond_lim, M)
    
    if is_spd(A) and cond_lim > cond(A):
        return matrix(A)
    else:
        print(f"Trial for {M} completed, cond = {cond(A)}")
        spd_pentadiag(M, cond_lim)
# ---------------Pentadiagonal SPD Matrix------------------







# Function signature
# def make_sparse_spd(M:int, alpha:float, l_val:float, u_val:float, cond_ulimit:int)->matrix:
E = {}

for i in range(n_cases):



    # M = 20 tridiagonal + spd
    if diag_mat[i] and M[i]==20:
        E.update({M[i]:spd_tridiag(M[i], cond_lim[i])})




    # For M = 40 non spd and Sparse
    if not spd[i] and not diag_mat[i]:
        for iter in range(1000):
            A = rand(M[i], M[i], density=1-sparsity[i]).todense()
            print(iter,"\n",cond(A))
            A = reduce_cond_num(A, cond_lim[i], M[i])
            print(cond(A))
            if cond(A) < cond_lim[i]:
                break

        E.update({M[i]:A})
        print(f"REACHED M = {M[i]} with cond = ",cond(E[M[i]]))




    # for M = 80, 160, 320 and sparse
    if spd[i] and not diag_mat[i]:
        
        if M[i] != 160:
            A = make_sparse_spd(M[i], sparsity[i], l_val=0.1, u_val=0.9)
        
        else:
            A = spd_pentadiag(M[i], cond_lim[i])
        
        for iter in range(1000):
            print(iter, "\t", cond(A))
            A = reduce_cond_num(A, cond_lim[i], M[i])
            print(M[i], "\t", cond(A))

            if cond(A) < cond_lim[i]:
                break

            else:
                # A = make_sparse_spd(M[i], sparsity[i], l_val=0.1, u_val=0.9)
                if M[i] != 160:
                    A = make_sparse_spd(M[i], sparsity[i], l_val=0.1, u_val=0.9)
                else:
                    A = spd_pentadiag(M[i], cond_lim[i])
                    print(f"TRIAL {i} for M = {M[i]}")


        if np.size(A) > 1:
            E.update({M[i]:A})
            print(f"REACHED M = {M[i]} with cond = ",cond(E[M[i]]))

        else:
            print(f"Error at M = {M[i]} and cond = {cond_lim[i]}")




    # M = 320 --> pentadiag + spd
    if diag_mat[i] and M[i]==320:
        E.update({f'{M[i]}_p':spd_pentadiag(M[i], cond_lim[i])})
        print(f"REACHED M = {M[i]} with cond = ",cond(E[f'{M[i]}_p']))
        
for key in E.keys():
    save_spd(E[key], f"A_{key}.txt")
