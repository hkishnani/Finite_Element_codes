#ifndef __BASIS_HPP

#define __BASIS_HPP
#include<eigen3/Eigen/Dense>
#include<eigen3/Eigen/Sparse>

using namespace std;
using namespace Eigen;

typedef double (*func)(double, double*);
typedef double (*dfunc)(func, double, double*);

typedef double (*f_RHS)(double);

// for linear basis function over element 0-1
void print_sparse(SparseMatrix<double>&);
double phi_L_0(double , double* );
double phi_L_1(double , double* );
double d_phi(func, double, double*);
double integrate_phi(func, double*, int);
Matrix2d integrate_phi_i_x_phi_j(double*, int);
Vector2d integrate_f_phi_elem(f_RHS, double*);

#endif