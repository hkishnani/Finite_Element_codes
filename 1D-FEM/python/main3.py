# -*- coding: utf-8 -*-

"""
Created on Sun Aug 25 03:10:18 2024

@author: himanshu
"""

# Solve -u" = f --> f = sin(x)

from numpy.linalg import solve
from numpy import zeros, array, linspace, polyfit, copy
from scipy.sparse import spdiags
from numpy import sin, cos, pi
from basis import integrate_phi_i_x_phi_j as int_ij
from basis import integrate_f_phi_elem as f_phi
from solvers import cholesky_decomposition,\
    forward_substitution, backward_substitution

#%% Post libraries
from post import plot_solution_1D, calc_L2_norm, plot_convergence


# %% problem specification
# eq ==> -u" = f
f = sin
u = sin

n_elem_list = [10, 20, 40, 80, 160]     # dof(s) = n_elem + 1
count = 0
L2_norm_list = []
h_list = []

x_domain = array([0, pi])


for n_elem in n_elem_list:
    u_D = array([0, n_elem])
    g_D = u(x_domain)

    A_G = zeros((n_elem+1, n_elem+1))
    b = zeros((n_elem+1))

    #%% Discretization
    x_node = linspace(start = x_domain[0], 
                      stop = x_domain[1], 
                      num = n_elem+1)
    
    x_elem_dom = zeros(2)
    
    for row in range(n_elem):
        A_G[row  , row+1] = -1/(x_node[row+1] - x_node[row  ])
        A_G[row  , row-1] = -1/(x_node[row  ] - x_node[row-1])
        A_G[row  , row  ] = - (A_G[row, row+1] + A_G[row, row-1])

        x_elem_dom[0] = x_node[row]
        x_elem_dom[1] = x_node[row+1]
        
        b[row:row+2] += f_phi(f, x_elem_dom)
        
    # implementation of Boundary conditions
    A_CG = A_G[1:-1,1:-1]
    b_CG = b[1:-1]
    
    #%% Solve using Cholesky
    A_test = cholesky_decomposition(copy(A_CG))
    x = forward_substitution(A_test.T, copy(b_CG))
    x = backward_substitution(A_test, x)
    print(x)
    u_h = zeros(n_elem+1)
    u_h[1:-1] = x

    #%% post processing data
    h_list.append((x_domain[1] - x_domain[0])/n_elem)
    L2_norm_list.append(calc_L2_norm(x_node, u_h, u))
    
    #%% Post
    plot_solution_1D(u_h, x_node, n_elem, u)
    
    count+=1

plot_convergence(h_list, L2_norm_list)
