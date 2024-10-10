#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 07:25:51 2024

@author: himanshu
"""

import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad
import plotly.io as pio
from math import log


def plot_solution_1D(zeta, x_node, n_elem, u):
    fig = go.Figure()
    for i in range(0, n_elem):
        x_dom = np.array([x_node[i], x_node[i+1]])
        y_dom = np.array([zeta[i]  , zeta[i+1]  ])

        # y = ax + b  --> piecewise solution
        a,b = np.polyfit(x_dom, y_dom, 1)
        u_h = lambda x: a*x+b

        fig.add_trace(
            go.Scatter(x=x_dom, y=u_h(x_dom),
                       mode='lines+markers+text',
                       name = f"element {i}_{i+1}",
                       text = [f"{round(x_dom[0],4)}, {round(y_dom[0],4)}",
                               f"{round(x_dom[1],4)}, {round(y_dom[1],4)}"]
                       )
            )

    fig.add_trace(go.Scatter(x = np.linspace(0, np.pi, 1000), 
                             y = u(np.linspace(0, np.pi, 1000)),
                             mode = 'lines')
                  )
    
    fig.update_traces(textposition='middle left', 
                      textfont_size=20)
    fig.show(renderer= "browser")
    
    
#%% Calculate L2 norm
def calc_L2_norm(x, phi, u):

    nelem = x.size - 1
    L2_norm = 0.0

    for i in range(0,nelem):
        x_dom = np.array([x[i], x[i+1]])
        y_dom = np.array([phi[i], phi[i+1]])
    
        # y = ax + b  --> piecewise solution
        a,b = np.polyfit(x_dom, y_dom, 1)
        u_h = lambda x: a*x+b
        
        err = lambda x: (u(x) - u_h(x))**2
        L2_norm += quad(err, x[i], x[i+1])[0]
    
    L2_norm = L2_norm**0.5
    return L2_norm


#%% Convergence Plot
def plot_convergence(h_list, L2_norm_list):
    # plot L2 error vs h
    fig = go.Figure()
    
    slope = [   (log(L2_norm_list[i]) - log(L2_norm_list[i+1]))
             /  (log(h_list[i])       - log(h_list[i+1]))
             for i in range(len(h_list)-1)
             ]
    
    val=0
    for s in slope:
        val += s
    val /= len(h_list)-1
    
    for h_i, L2_err_i in zip(h_list, L2_norm_list):
        print(h_i, L2_err_i)
        fig.add_trace(go.Scatter(x = [h_i], y = [L2_err_i],
                                 mode ='markers+text',
                                 text = f"{round(h_i,4)}, {round(L2_err_i,6)}",
                                 textposition='bottom center'
                                 ))
        fig.update_layout(showlegend=False)

    fig.add_trace(go.Scatter(x = h_list, y = L2_norm_list,
                             mode ='lines+text',
                             text=[f"avg slope = {round(val,4)}"],
                             textposition="top center"
                             )
                  )

    # inclined line
    x_slope = [  h_list[0]**(i)   for i in range(1, len(h_list) ) ]
    y_slope = [  h_list[0]**(2*i) for i in range(1, len(h_list) ) ]
    
    fig.add_trace(go.Scatter(x = x_slope, y = y_slope,
                             mode = 'lines+text',
                             line = dict(color='firebrick', width=4, 
                                         dash='dash'),
                             text = ["slope = 2"],
                             textposition='top center'
                             )
                  )

    fig.update_layout(
        title="L2 norm vs h",
        xaxis_title="h = 1/nelem",
        yaxis_title="L2_norm",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )


    fig.update_layout(
    yaxis = dict(
        showexponent = 'all',
        exponentformat = 'power'),
    xaxis = dict(
        showexponent = 'all',
        exponentformat = 'power')
    )

    fig.update_yaxes(type="log")
    fig.update_xaxes(type="log")
    fig.update_traces(textfont_size=15)

    fig.show()
    pio.write_image(fig, file="convergence_plot.svg", format='svg', scale=None,
                    width=1920, height= 1080, validate=True, engine='auto')