#include"basis.hpp"
#include<iostream>
#include<eigen3/Eigen/Sparse>
#include<eigen3/Eigen/Dense>
#include<fstream>

const double PI = 3.14159265;

using namespace std;
using namespace Eigen;

inline double f (double x) {return cos(x);}

int main()
{
    double x_domain[2] = {0.0, PI};
    int n_elem = 10;

    // Boundary Conditions
    Vector2i u_D;   u_D << 0, n_elem;   // node where BC. is applied
    Vector2d g_D;   g_D << 1.0, -1.0;   // val at that co-ord.
    // Boundary Conditions

    VectorXd x_node=VectorXd::LinSpaced(n_elem+1,x_domain[0],x_domain[1]);

    // Here is the equation A_G zeta = b
    SparseMatrix<double> A_G(n_elem+1, n_elem+1);
    VectorXd b(n_elem+1);
    // Here is the equation A_G zeta = b

    // ------Assembly of Stiffness Matrix & preparing RHS vector--------
    A_G.reserve(3*(n_elem+1));
    double x_elem_dom[2] = {0.0, 0.0};
    for (int row = 0; row < n_elem; row++)
    {
        x_elem_dom[0] = x_node(row); x_elem_dom[1] = x_node(row+1);

        // Evaluating local stiffness matrix
        Matrix2d A_L = integrate_phi_i_x_phi_j(x_elem_dom, 1);
        A_G.coeffRef(row  , row  ) += A_L(0,0);
        A_G.coeffRef(row  , row+1) += A_L(0,1);
        A_G.coeffRef(row+1, row  ) += A_L(1,0);
        A_G.coeffRef(row+1, row+1) += A_L(1,1);

        // RHS
        b({row,row+1}) += integrate_f_phi_elem(f, x_elem_dom);
    }
    // ------Assembly of Stiffness Matrix & preparing RHS vector--------

    // Imposing Dirichlet Boundary condition at node 0 and last
    for (int i = 0; i < u_D.size(); i++)
    {
        A_G.row(u_D(i)) *= 0;
        A_G.coeffRef(u_D(i),u_D(i)) += 1;
        b(u_D(i)) = g_D(i);
    }
    A_G.makeCompressed();   // Remove Extra zeros


    std::cout << "A_G = \n" << MatrixXd(A_G) << endl;
    std::cout << "b = \n" << b << endl;

    // -----------------------Solving A zeta = b------------------------
    VectorXd zeta = VectorXd::Zero(n_elem+1);
    BiCGSTAB<SparseMatrix <double, RowMajor> > solver;
    solver.compute(A_G);
    zeta = solver.solve(b);
    // -----------------------Solving A zeta = b------------------------

    ofstream myfile;
    myfile.open ("u_h.txt");
    for (int i = 0; i < x_node.size(); i++)
        myfile << x_node(i) << ',' << zeta(i) << endl;
    myfile.close();

    return 0;
}