#include<iostream>
#include<eigen3/Eigen/Dense>
#include "basis.hpp"
using namespace std;
using namespace Eigen;

// typedef double (*func)(double, double*);
// typedef double (*dfunc)(func, double, double*);

/*  ---------------  1
         \  /
          \/ 
          /\
    _____/__\______  0
       0      1
*/


// ---------------------------------------------------------------------
// First Test function

/// @brief Prints a sparsematrix... just a test function for this file
/// @param S_mat 
void print_sparse(SparseMatrix<double>& S_mat)
{
    cout << "\n" << MatrixXd(S_mat) << "\n" << endl;
}
// ---------------------------------------------------------------------


// phi linear which is 1 at node 0

/// @brief Linear basis function that is 1 @ node 0 and 0 @ node 1
/// @param x : point at which phi_0 needs to be evaluated
/// @param x_lim : Domain of x
/// @return : value of a piecewise linear function that is 1 at node 0
double phi_L_0( double x, double* x_lim )
{
    double result = (x_lim[1] - x)/(x_lim[1] - x_lim[0]);
    return (result>=0 && result <=1) ? result : 0 ;
}

// phi linear which is 1 at node 1

/// @brief Linear basis function that is 1 @ node 1 and 0 @ node 0
/// @param x : point at which phi_1 needs to be evaluated
/// @param x_lim : Domain of x
/// @return : value of a piecewise linear function that is 1 at node 1
double phi_L_1( double x, double* x_lim )
{
    double result = (x - x_lim[0])/(x_lim[1] - x_lim[0]);
    return (result>=0 && result <=1) ? result : 0 ;
    // Alternatively we can write (x_lim[1]-x)*(x-x_lim[0])>0
}

// Need to generalize this function for recursive call (Autodiff)
/// @brief This function calculates df/dx using gradient method
/// @param f : function pointer with typedef func
/// @param x : evaluation point
/// @param x_lim : x domain
/// @return : [ f(x+h) - f(x-h) ] / 2h
double d_phi( func f, double x, double* x_lim )
{
    double result = 0.0, h = 1e-6;
    double k = (x<x_lim[1] && x>=x_lim[0]) ? h : 0;
    double l = (x>x_lim[0] && x<=x_lim[1]) ? h : 0;
    double Dr = ((k+l) > 0) ? (k+l) : h;
    result = (f(x+k, x_lim) - f(x-l, x_lim)) / Dr;

    return result;
}

// Area under the nth order derivative of the curve
/// @brief integration of n th (0th or 1st) order differential of phi
/// @param f_o : the function phi w
/// @param x_lim : domain of integration
/// @param order : 0 for integration of phi or 1 for integration of phi'
/// @return : ∫ φi dx @ order = 1 or ∫ φi' dx @ order = 0(simpson's 1/3)
double integrate_phi( func f_o, double* x_lim, int order )
{
    // Generalization of function required
    double h = x_lim[1] - x_lim[0];
    double m = 0.5*(x_lim[1] + x_lim[0]);
    double PHI_0, PHI_M, PHI_1;

    if (order == 0)
    {
        PHI_0 = (*f_o)(x_lim[0], x_lim);
        PHI_M = (*f_o)(m       , x_lim);
        PHI_1 = (*f_o)(x_lim[1], x_lim);
    }

    if (order == 1)
    {
        PHI_0 = d_phi(f_o, x_lim[0], x_lim);
        PHI_M = d_phi(f_o, m       , x_lim);
        PHI_1 = d_phi(f_o, x_lim[1], x_lim);
    }

    return (PHI_0 + 4*PHI_M + PHI_1)*(h/6);
}


/// @brief < ∫φi φj dx @ order 0 > or < ∫φi' φj' dx @ order 1 >
/// @param x_lim : range of x for which integration is valid
/// @param order : changes integrand function. φ @ 0 or φ' @ 1
/// @return : Integration evaluated using Simpson's 1/3 rule
Matrix2d integrate_phi_i_x_phi_j(double* x_lim, int order )
{
    // Generalization of this function is required
    double h = x_lim[1] - x_lim[0];
    double m = 0.5*(x_lim[1] + x_lim[0]);
    double PHI_0[2], PHI_M[2], PHI_1[2];

    Matrix2d A{{0.0, 0.0},
               {0.0, 0.0}};

    // do ∫ (φi * φj) dx
    // find a smarter trick in this one...
    if (order == 0)
    {
        PHI_0[0] = phi_L_0(x_lim[0] , x_lim);
        PHI_M[0] = phi_L_0(m        , x_lim);
        PHI_1[0] = phi_L_0(x_lim[1] , x_lim);

        PHI_0[1] = phi_L_1(x_lim[0] , x_lim);
        PHI_M[1] = phi_L_1(m        , x_lim);
        PHI_1[1] = phi_L_1(x_lim[1] , x_lim);
    }

    // do ∫ (φi' * φj') dx
    if (order == 1)
    {
        PHI_0[0] = d_phi(phi_L_0, x_lim[0], x_lim );
        PHI_M[0] = d_phi(phi_L_0, m       , x_lim );
        PHI_1[0] = d_phi(phi_L_0, x_lim[1], x_lim );

        PHI_0[1] = d_phi(phi_L_1, x_lim[0], x_lim );
        PHI_M[1] = d_phi(phi_L_1, m       , x_lim );
        PHI_1[1] = d_phi(phi_L_1, x_lim[1], x_lim );
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            A(i,j) =(   PHI_0[i]*PHI_0[j]
                     +4*PHI_M[i]*PHI_M[j]
                     +  PHI_1[i]*PHI_1[j]
                    )*(h/6);
        }
        // Using Simpson's 1/3rd rule
    }
    return A;
}
// i = 0 ==> f = phi_L_0
// i = 1 ==> f = phi_L_1
/*
even though phi and d_phi have different signature
*/


// Needs to be called in a loop
Vector2d integrate_f_phi_elem(f_RHS f, double* x_elem)
{
    double m = (x_elem[0] + x_elem[1])*0.5;
    Vector2d vec; vec(0)=0.0; vec(1)=0.0;
    double h = x_elem[1] - x_elem[0];
    double PHI_0[2], PHI_M[2], PHI_1[2];
    PHI_0[0] = phi_L_0(x_elem[0] , x_elem);
    PHI_M[0] = phi_L_0(m         , x_elem);
    PHI_1[0] = phi_L_0(x_elem[1] , x_elem);

    PHI_0[1] = phi_L_1(x_elem[0] , x_elem);
    PHI_M[1] = phi_L_1(m         , x_elem);
    PHI_1[1] = phi_L_1(x_elem[1] , x_elem);

    for(int i=0; i<2; i++)
    {
        vec(i) = (    PHI_0[i]*f(x_elem[0])
                +   4*PHI_M[i]*f(m)
                +     PHI_1[i]*f(x_elem[1])
                ) * (h/6);
    }
    return vec;
}
