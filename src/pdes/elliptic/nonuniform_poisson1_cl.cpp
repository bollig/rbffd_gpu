//#define VIENNACL_HAVE_UBLAS 1
#include <stdlib.h>
#include <math.h>

#include "grids/grid.h"
#include "nonuniform_poisson1_cl.h"
#include "exact_solutions/exact_solution.h"
#include "timingGE.h"

// The GPU/OpenCL side sparse arrays come from ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/io/matrix_market.hpp"

#include <iostream>
#include <fstream>
#include <vector>

// The CPU Side sparse arrays come from Boost uBlas.
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
using namespace std;

NonUniformPoisson1_CL::NonUniformPoisson1_CL(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_) :
        NCARPoisson1(_solution, subdomain_, der_, rank, dim_num_)
{}

NonUniformPoisson1_CL::NonUniformPoisson1_CL(ProjectSettings* _settings, ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_) :
        NCARPoisson1(_settings, _solution, subdomain_, der_, rank, dim_num_)
{}

//----------------------------------------------------------------------

NonUniformPoisson1_CL::~NonUniformPoisson1_CL() {
}
//----------------------------------------------------------------------
// Solve the poisson system.
// NOTE: this routine is old and uses a possibly incorrect method for solving with the
// neumann boundary conditions. I am starting an alternate routine to solve the system
// in the same fashion that Joe solved the system.
void NonUniformPoisson1_CL::solve(Communicator* comm_unit) {
    t1.start();
    if (subdomain == NULL) {
        cerr
                << "In " << __FILE__
                << " No GPU class passed to Constructor. Cannot perform intermediate communication/updates in solver."
                << endl;
        exit(EXIT_FAILURE);
    } else {

        int nb = subdomain->global_boundary_nodes.size();
        // All interior and boundary nodes are included in the stencils.
        // The first nb Q_stencils should be the global boundary nodes
        int ni = subdomain->Q_stencils.size() - nb;

        int nn = (nb + ni) ;

        // TODO: solve this in parallel
        //     comm_unit->broadcastObjectUpdates(subdomain);

        // Do NOT use GPU as buffer for computation
        // Only go up to the number of stencils since we solve for a subset of the values in U_G
        // Since U_G in R is at end of U_G vector we can ignore those.
        //for (int i = 0; i < s.size(); i++) {
        //    s[i] = subdomain->U_G[i];
        //}

        //for (int i = 0; i < lapl_deriv.size(); i++) {
        //  Vec3& v = (*rbf_centers)[i];
        //    printf("(local: %d), lapl(%f,%f,%f)= %f\t%f\n", i, v.x(), v.y(), v.z(),
        //            lapl_deriv[i], s[i]);
        //}

        // Evan TODO:
        //
        // 1) Build a sparse matrix representation for all the interior derivative weights LA
        // 2) Build a full vector F = laplacian(u)
        // 3)

        // We are forming:
        //
        //  [ w_ddr -1/r*I  ]  [ A_boundary ]     [ 0  ]
        //  [ w_lapl    ]  [ A_interior ]  =  [ f_interior ]
        //
        // where w_lapl are the laplacian RBFFD weights for interior nodes (ni x nb + ni)
        // w_dr are the d/dr RBFFD weights (that is, the operator xd/dx +
        //

        // The w_lapl, weights for the laplacian, require (d^2 Phi / dx^2 + d^2 Phi / dy^2 + d^2 Phi / dz^2).
        // For w_ddr, weights for dA/dr, require (x*dPhi/dx + y*dPhi/dy + z*dPhi/dz). That means we will need to get
        // the stencil centers (Vec3) into the

        t2.start();
        int numNonZeros = 0;
#if 1
        // Compute all weights including those for the boundary nodes
        for (int i = 0; i < nb + ni; i++) {
            //subdomain->printStencil(subdomain->Q_stencils[i], "Q[i]");
            // Compute all derivatives for our centers and return the number of
            // weights that will be available
            numNonZeros += der->computeWeights(subdomain->G_centers, subdomain->Q_stencils[i], i, dim_num);
            der->computeWeightsSVD(subdomain->G_centers, subdomain->Q_stencils[i], i, "x");
            der->computeWeightsSVD(subdomain->G_centers, subdomain->Q_stencils[i], i, "y");
            der->computeWeightsSVD(subdomain->G_centers, subdomain->Q_stencils[i], i, "lapl");
        }
#else
        // Compute only interior weights
        for (int i = 0; i < nb; i++) {
            // 1 nonzero per row for boundary (dirichlet has I since we know values and dont need weights)
            //der->computeWeights(subdomain->G_centers, subdomain->Q_stencils[i], subdomain->Q_stencils[i][0], dim_num);
            numNonZeros += 1;
        }
        for (int i = nb; i < nb + ni; i++) {
            //subdomain->printStencil(subdomain->Q_stencils[i], "Q[i]");
            // Compute all derivatives for our centers and return the number of
            // weights that will be available
            numNonZeros += der->computeWeights(subdomain->G_centers, subdomain->Q_stencils[i], subdomain->Q_stencils[i][0], dim_num);
        }
#endif
        t2.end();
        cout << "Weights computed" << endl;

        t3.start();

        int nm = nb + ni;
        if ((boundary_condition != DIRICHLET) && (!disable_sol_constraint)) {
            //nm = nm+1; // Constrain constant on far right of the implicit system
             nm = nm+4; // Constrain constant, plus X and Y on far right of implicit system
        }

        // 1) Fill the ublas matrix on the CPU
        //boost::numeric::ublas::compressed_matrix<FLOAT> L_host(nm, nm);
        //boost::numeric::ublas::vector<FLOAT> F_host(nm);
        MatType L_host(nm, nm);
        VecType F_host(nm);

        F_host(nm-1) = 0.;

        int indx = 0;

        //--------------------------------------------------
        // Fill Boundary weights (LHS + RHS)
        if (test_dirichlet_lockdown) {
            cout << "TESTING DIRICHLET LOCKDOWN MODE 3" << endl;
            // Test the case when we specify one Dirichlet point on each boundary. Does this help us tie down the solution? No.
            // What about when we specify 3 points on the boundaries (i.e. a triangle to give us an orientation)?
            indx += this->fillBoundaryDirichlet(L_host, F_host, subdomain->Q_stencils[0], subdomain->G_centers);
            //indx += this->fillBoundaryDirichlet(L_host, F_host, subdomain->Q_stencils[1], subdomain->G_centers);
            //indx += this->fillBoundaryDirichlet(L_host, F_host, subdomain->Q_stencils[2], subdomain->G_centers);
            //for (int i = 3; i < nb-3; i++) {
            for (int i = 1; i < nb-2; i++) {
                switch (boundary_condition) {

                case DIRICHLET:
                    indx += this->fillBoundaryDirichlet(L_host, F_host, subdomain->Q_stencils[i], subdomain->G_centers);
                    break;

                case NEUMANN:
                    indx += this->fillBoundaryNeumann(L_host, F_host, subdomain->Q_stencils[i], subdomain->G_centers);
                    break;

                case ROBIN:
                    indx += this->fillBoundaryRobin(L_host, F_host, subdomain->Q_stencils[i], subdomain->G_centers);
                    break;

                default:
                    cout << "ERROR! boundary_condition has invalid value: " << boundary_condition << endl;
                    exit(EXIT_FAILURE);
                }
            }
            //indx += this->fillBoundaryDirichlet(L_host, F_host, subdomain->Q_stencils[nb-3], subdomain->G_centers);
            indx += this->fillBoundaryDirichlet(L_host, F_host, subdomain->Q_stencils[nb-2], subdomain->G_centers);
            indx += this->fillBoundaryDirichlet(L_host, F_host, subdomain->Q_stencils[nb-1], subdomain->G_centers);
        } else {
            cout << "NOT USING DIRICHLET LOCKDOWN MODE" << endl;
            // Normal fill with homogenous boundary conditions. The solutions are not tied down very well.
            for (int i = 0; i < nb; i++) {
                switch (boundary_condition) {

                case DIRICHLET:
                    indx += this->fillBoundaryDirichlet(L_host, F_host, subdomain->Q_stencils[i], subdomain->G_centers);
                    break;

                case NEUMANN:
                    indx += this->fillBoundaryNeumann(L_host, F_host, subdomain->Q_stencils[i], subdomain->G_centers);
                    break;

                case ROBIN:
                    indx += this->fillBoundaryRobin(L_host, F_host, subdomain->Q_stencils[i], subdomain->G_centers);
                    break;

                default:
                    cout << "ERROR! boundary_condition has invalid value: " << boundary_condition << endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
        cout << "DONE FILLING BOUNDARY" << endl;

        //--------------------------------------------------
        // Fill Interior weights (LHS + RHS)
        for (int i = nb; i < nb+ni; i++) {
            indx += this->fillInterior(L_host, F_host, subdomain->Q_stencils[i], subdomain->G_centers);
        }

        cout << "DONE FILLING INTERIOR" << endl;

        //--------------------------------------------------
        // Fill Additional Constraint weights (LHS + RHS)
        if ((boundary_condition != DIRICHLET) && (!disable_sol_constraint)) {
            // I forget: is this required for BOTH neumann and robin?
            this->fillSolutionConstraint(L_host, F_host, subdomain->Q_stencils, subdomain->G_centers, nb, ni);
            cout << "DONE ADDING CONSTRAINT" << endl;
        }

        //--------------------------------------------------
        // Solve and post-process

        if ((indx - numNonZeros) != 0) {
            cerr << "WARNING! HOST MATRIX WAS NOT FILLED CORRECTLY. DISCREPANCY OF " << (indx - numNonZeros) << " NONZERO ELEMENTS!" << endl;
            //     exit(EXIT_FAILURE);
        }
        t3.end();
        cout << "Implicit system assembled" << endl;
        viennacl::io::write_matrix_market_file(L_host, "L_host.mtx");

        // cout << "L_HOST: " << L_host.size1() << "\t" << L_host.size2() << "\t" << L_host.filled1() << "\t" << L_host.filled2() << endl;
        // cout << L_host << endl;
        // cout << "F: " << F_host << endl;

        if (check_L_p_Lt) {
            // viennacl::transposed_matrix_proxy<FLOAT, 1, > L_transp_host(L_host);
            L_host = L_host + trans(L_host);
        }

        // 2) Convert to OpenCL space:

        viennacl::compressed_matrix<FLOAT, 1 /*Alignment(e.g.: 1,4,8)*/ > L_device(L_host.size1(), L_host.size2());
        viennacl::vector<FLOAT> F_device(F_host.size());
        viennacl::vector<FLOAT> x_device(F_host.size());

        boost::numeric::ublas::vector<FLOAT> x_host(F_host.size());

        t4.start();
        cout << "Before copy to GPU" << endl;
        // copy to GPU
        copy(L_host, L_device);
        copy(F_host, F_device);

        cout << "Solving system" <<endl;
        t5.start();
        //x_device = viennacl::linalg::prod(L_device, F_device);
        x_device = viennacl::linalg::solve(L_device, F_device, viennacl::linalg::bicgstab_tag(1.e-24, 3000));

        viennacl::ocl::finish();
        t5.end();

        cout << "Done with solve" << endl;

        // Copy solution to host
        copy(x_device, x_host);

        t4.end();

        cout << "Results copied to host" << endl;

        boost::numeric::ublas::vector<FLOAT> exact_host(F_host.size());
        boost::numeric::ublas::vector<FLOAT> error_host(F_host.size());
        boost::numeric::ublas::vector<FLOAT> rel_error_host(F_host.size());

        for (int i = 0; i < nb + ni; i++) {
            exact_host(subdomain->Q_stencils[i][0]) = (FLOAT) (exactSolution->at(subdomain->G_centers[subdomain->Q_stencils[i][0]], 0.));
        }
        for (int i = nb+ni; i < exact_host.size(); i++) {
            exact_host(i) = 0.;
        }
#if 1
#define CATCH_ZERO_RELATIVE_ERROR 0
        //error_host = x_host - exact_host;
        for (int i = 0; i < nb+ni; i++) {
            error_host[i] = fabs(x_host[i] - exact_host[i]);
#if CATCH_ZERO_RELATIVE_ERROR
            if ((exact_host[i] < 1e-10) && (error_host[i] < 1e-10)) {
                rel_error_host[i] = 0.;
            } else {
                rel_error_host[i] = error_host[i] / fabs(exact_host[i]);
            }
#else
            rel_error_host[i] = error_host[i] / fabs(exact_host[i]);
#endif

        }
#endif

        for (int i = nb+ni; i < error_host.size(); i++) {
            rel_error_host[i] = 0.;
        }

        boost::numeric::ublas::vector<FLOAT> residual = prod(L_host, exact_host) - F_host;

        cout << "Writing results to disk" << endl;

        ofstream fout;
        //fout.open("L.mtx");
        //fout << L_host << endl;
        //fout.close();
        viennacl::io::write_matrix_market_file(L_host, "L.mtx");

        this->write_to_file(F_host, "F.mtx");
        this->write_to_file(exact_host, "X_exact.mtx");
        this->write_to_file(x_host, "X_approx.mtx");
        this->write_to_file(residual, "R.mtx");
        this->write_to_file(error_host, "E_absolute.mtx");
        this->write_to_file(rel_error_host, "E_relative.mtx");

        std::cout << "Relative residual || x_exact - x_approx ||_2 / || x_exact ||_2  = " << norm_2(exact_host - x_host) / norm_2(exact_host) << std::endl;
        std::cout << "Relative residual || A*x_exact - F ||_2 / || F ||_2  = " << norm_2(prod(L_host, exact_host) - F_host) / norm_2(F_host) << std::endl;
        std::cout << "Relative residual || A*x_approx - F ||_2 / || F ||_2  = " << norm_2(prod(L_host, x_host) - F_host) / norm_2(F_host) << std::endl;
        std::cout << "[Precision] sizeof(FLOAT) = " << sizeof(FLOAT) << " bytes (4 = single; 8 = double)" << std::endl;

        cout.flush();
    }


    t1.end();

    return;
}


int NonUniformPoisson1_CL::fillBoundaryNeumann(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers)
{
    int indx = 0;


    double* x_weights = der->getXWeights(stencil[0]);
    double* y_weights = der->getYWeights(stencil[0]);
    double* z_weights = der->getZWeights(stencil[0]);

    // DONT FORGET TO ADD IN THE -1/r on the stencil center weights
    Vec3& center = centers[stencil[0]];
    double r = center.magnitude();

    if (r < 1e-8) {
        cerr << "WARNING! VANISHING SPHERE RADIUS! CANNOT FILL -1/r in " << __FILE__ << endl;
        exit(EXIT_FAILURE);
    }

    //--------- SETUP LHS SYSTEM BASED ON BOUNDARY CONDITIONS -----------

    // Neumann condition says:  dPhi(x)/dn = f(x)
    //      dPhi/dn = Grad(Phi(x)) .dot. n(x)
    // where n(x) is the UNIT normal vector n (NORMALIZED!!! WITH PROPER SIGN!!!!)
    //

    Vec3 normal = center;
    normal.normalize();

    if (r < 0.6) {
        cout << "STENCIL " << stencil[0] << " IS INNER BOUNDARY: switching normal sign" << endl;
        normal *= -1;
    }

    for (int j = 0; j < stencil.size(); j++) {

        double value = (normal.x() * x_weights[j] + normal.y() * y_weights[j] + normal.z() * z_weights[j]);
        // When j == 0 we should have i = Q_stencil[i][j] (i.e., its the center element
        L(stencil[0],stencil[j]) = (FLOAT)value;
        indx++;
    }

    this->fillBoundaryNeumannRHS(F, stencil, centers);

    return indx;
}

void NonUniformPoisson1_CL::fillBoundaryNeumannRHS(VecType& F, StencilType& stencil, CenterListType& centers)
{
    double rhs_val;
    double* x_weights = der->getXWeights(stencil[0]);
    double* y_weights = der->getYWeights(stencil[0]);
    double* z_weights = der->getZWeights(stencil[0]);
    Vec3& center = centers[stencil[0]];
    Vec3 normal = center;
    normal.normalize();

    if (this->use_discrete_rhs) {

        // Discrete Neuman
        //F_host(i) = (FLOAT)0.;
        double discrete_condition = 0.;

        for (int j = 0; j < stencil.size(); j++) {
            double weight = (normal.x() * x_weights[j] + normal.y() * y_weights[j] + normal.z() * z_weights[j]);
            Vec3& vj = centers[stencil[j]];
            discrete_condition += weight * exactSolution->at(vj,0);  //   laplacian(vj.x(), vj.y(), vj.z(), 0.));
        }
        rhs_val = discrete_condition;
    } else {

        // Continuous: n * (d/dx + d/dy + d/dz) = ?   (value determined by exact solution's analytic derivatives)
        rhs_val = (normal.x() *exactSolution->xderiv(center,0.) + normal.y() *exactSolution->yderiv(center,0.) + normal.z() *exactSolution->zderiv(center,0.));
    }

    F[stencil[0]] = (FLOAT) rhs_val;
}


int NonUniformPoisson1_CL::fillBoundaryRobin(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers)
{
    int indx = 0;

    double* x_weights = der->getXWeights(stencil[0]);
    double* y_weights = der->getYWeights(stencil[0]);
    double* z_weights = der->getZWeights(stencil[0]);

    // DONT FORGET TO ADD IN THE -1/r on the stencil center weights
    Vec3& center = centers[stencil[0]];
    double r = center.magnitude();

    if (r < 1e-8) {
        cerr << "WARNING! VANISHING SPHERE RADIUS! CANNOT FILL -1/r in " << __FILE__ << endl;
        exit(EXIT_FAILURE);
    }

    Vec3 normal = center;
    normal.normalize();

    if (r < 0.6) {
        cout << "STENCIL " << stencil[0] << " IS INNER BOUNDARY: switching normal sign" << endl;
        normal *= -1;
    }

    for (int j = 0; j < stencil.size(); j++) {

        // The Robin boundary condition says:
        //      n .dot. d(Phi/r)/dr = f
        // where n is the UNIT normal
        // Use the quotient rule to reduce this to:
        //  ( n .dot. d(Phi)/dr ) * (1/r) - Phi/(r^2) = f
        // Then we approximate the (n .dot. d(Phi/dr)) with:
        double weight = (normal.x() * x_weights[j] + normal.y() * y_weights[j] + normal.z() * z_weights[j]) / r;

        // Remember to remove 1/r^2 for the boundary condition: n d/dr(a/r) = 0
        if (j == 0) {
            weight -= (1./(r*r));
        }
        // When j == 0 we should have i = Q_stencil[i][j] (i.e., its the center element
        L(stencil[0],stencil[j]) = (FLOAT)weight;
        indx++;
    }

    this->fillBoundaryRobinRHS(F, stencil, centers);

    return indx;
}


void NonUniformPoisson1_CL::fillBoundaryRobinRHS(VecType& F, StencilType& stencil, CenterListType& centers)
{
    double rhs_val;
    double* x_weights = der->getXWeights(stencil[0]);
    double* y_weights = der->getYWeights(stencil[0]);
    double* z_weights = der->getZWeights(stencil[0]);
    Vec3& center = centers[stencil[0]];
    double r = center.magnitude();
    Vec3 normal = center;
    normal.normalize();

    if (this->use_discrete_rhs) {

        // Discrete Robin
        double discrete_condition = 0.;
        for (int j = 0; j < stencil.size(); j++) {
            // [(x*d/dx + y*d/dy + z*d/dz)(1/r) - (1/r^2)]*(\Phi) {Handle the 1/r^2 below}
            //
            // Use quotient rule:
            // n .dot. d/dr(A/r) = (n .dot. dA/dr)*(1/r) - A/(r*r)
            // = (n.x * dA/dx + n.y + dA/dy + n.z * dA/dz) * (1/r) - A/(r*r)
            // NOTE: handle all to left of minus sign here:
            double weight = (normal.x() * x_weights[j] + normal.y() * y_weights[j] + normal.z() * z_weights[j]) / r;
            Vec3& vj = centers[stencil[j]];
            discrete_condition += weight * exactSolution->at(vj,0);
        }
        // Handle the (-A/(r*r)) here
        discrete_condition -= exactSolution->at(center,0) / (r*r);

        rhs_val = discrete_condition;
    } else {

        // Continuous: (n.x * dA/dx + n.y + dA/dy + n.z * dA/dz) * (1/r) - A/(r*r) = ?   (value determined by exact solution's analytic derivatives)
        double n_dot_dadr = (normal.x() *exactSolution->xderiv(center,0.) + normal.y() *exactSolution->yderiv(center,0.) + normal.z() *exactSolution->zderiv(center,0.));
        rhs_val = (n_dot_dadr / r) - exactSolution->at(center,0.) / (r*r);
    }

    F[stencil[0]] = (FLOAT) rhs_val;
}


int NonUniformPoisson1_CL::fillBoundaryDirichlet(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers)
{
    int indx = 0;

    //--------- BOUNDARY NODE ENTRIES -----------
    for (int j = 0; j < stencil.size(); j++) {
        if (j == 0) {
            L(stencil[0],stencil[j]) = (FLOAT)1.;
        } else {
            L(stencil[0],stencil[j]) = (FLOAT)0.;
        }
        indx++;
    }

    F[stencil[0]] = (FLOAT) 0.;

    return indx;
}

void NonUniformPoisson1_CL::fillSolutionConstraint(MatType& L, VecType& F, StencilListType& stencils, CenterListType& centers, int nb, int ni)
{
    int nn = nb + ni;
    // constraint on solution

    F(nn) = 0.0;    // Constant value
    F(nn+1) = 0.0;  // X-coeff value
    F(nn+2) = 0.0;  // Y-coeff value
    F(nn+3) = 0.0;  // FOR SOL_CONSTRAINT

// Constrain solution constraint only the boundary
    for (int i = 0; i < nb; i++) {
        // last columns
        L(stencils[i][0], nn) = 1;  // Constrain constant
        L(stencils[i][0], nn+1) = centers[stencils[i][0]].x();  // Constrain X
        L(stencils[i][0], nn+2) = centers[stencils[i][0]].y();  // Constrain Y

        // SOL_CONSTRAINT: solution constraint to make the Neumann boundary condition complete
        // last row
        L(nn+3, stencils[i][0]) = 1;


        // update of RHS
        Vec3& v = centers[stencils[i][0]];

        // FOR SOL_CONSTRAINT last element of last row and last column: sum of exact solutions at all points
        F(nn+3) += exactSolution->at(v,0);
    }

// Ignore solution constraint on the interior
    for (int i = nb; i < nb+ni; i++) {
        // last columns
        L(stencils[i][0], nn) = 1;  // Constrain constant
        L(stencils[i][0], nn+1) = centers[stencils[i][0]].x();  // Constrain X
        L(stencils[i][0], nn+2) = centers[stencils[i][0]].y();  // Constrain Y
    }

#if 0
    for (int i = 0; i < nb+ni; i++) {
        // last columns
        L(stencils[i][0], nn) = 1;  // Constrain constant
        L(stencils[i][0], nn+1) = centers[stencils[i][0]].x();  // Constrain X
        L(stencils[i][0], nn+2) = centers[stencils[i][0]].y();  // Constrain Y

        // SOL_CONSTRAINT: solution constraint to make the Neumann boundary condition complete
        // last row
        L(nn+3, stencils[i][0]) = 1;


        // update of RHS
        Vec3& v = centers[stencils[i][0]];

        // FOR SOL_CONSTRAINT last element of last row and last column: sum of exact solutions at all points
        F(nn+3) += exactSolution->at(v,0);
    }
#endif

    // Constrain by specifying exactly what the value of the aX + bY + c = f coeffs are
    L(nn,nn) = 1.0;
    L(nn+1,nn+1) = 1.0;
    L(nn+2,nn+2) = 1.0;

    L(nn+3,nn+3) = 0.0;     // FOR SOL_CONSTRAINT

    L(nn+2,nn+3) = 1.0;     // FOR SOL_CONSTRAINT because we have one whole colum of zeros, so we fill in with a 1 to say that
    F(nn+2) = F(nn+3);      // The sum of SOL_CONSTRAINT and the coeff of this entry are equal to the sum of the sol constraint (redundant I know)
}

#if 0
void NonUniformPoisson1_CL::fillSolutionConstraint(MatType& L, VecType& F, StencilListType& stencils, CenterListType& centers, int nb, int ni)
{
    int nn = nb + ni;
    // constraint on solution

    F(nn) = 0.0;
    for (int i = 0; i < nb+ni; i++) {
        // last row
        L(nn, stencils[i][0]) = 1;
        // last column
        L(stencils[i][0], nn) = 1;  // Constrain constant
        // Final constraint to make the Neumann boundary condition complete

        // update of RHS
        Vec3& v = centers[stencils[i][0]];

        // last element of last row and last column: sum of exact solutions at all points
        F(nn) += exactSolution->at(v,0);
    }
    // No constraint on bottom right corner
    L(nn,nn) = 0.0;
}
#endif


void NonUniformPoisson1_CL::fillSolutionNoConstraint(MatType& L, VecType& F, StencilListType& stencils, CenterListType& centers, int nb, int ni)
{
    int nn = nb+ni;
    // constraint on solution
    F(nn) = 0.0;
    for (int i = 0; i < nb+ni; i++) {
        // last row
        L(nn, stencils[i][0]) = 0;
        // last column
        L(stencils[i][0], nn) = 0;
    }
    // No constraint on bottom right corner
    L(nn,nn) = 1;
}


int NonUniformPoisson1_CL::fillInterior(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers)
{
    int indx = 0;
    // Count only the interior fill
    indx += this->fillInteriorLHS(L, F, stencil, centers);
    this->fillInteriorRHS(L, F, stencil, centers);

    return indx;
}


int NonUniformPoisson1_CL::fillInteriorLHS(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers)
{
    int indx = 0;
    Vec3 st_center = centers[stencil[0]];
    // Do I want the gradK at each stencil node or only the center?
    double diffusivity = exactSolution->diffuseCoefficient(st_center);
    Vec3* gradK = exactSolution->diffuseGradient(st_center);

    //--------------------------------------------------
    // Fill Interior weights (LHS)
    if (use_uniform_diffusivity) {
        double* lapl_weights = der->getLaplWeights(stencil[0]);
        for (int j = 0; j < stencil.size(); j++) {
            L(stencil[0],stencil[j]) = (FLOAT) lapl_weights[j];
            //cout << "lapl_weights[" << j << "] = " << lapl_weights[j] << endl;
            indx++;
        }
    } else {
        double* x_weights = der->getXWeights(stencil[0]);
        double* y_weights = der->getYWeights(stencil[0]);
        double* z_weights = der->getZWeights(stencil[0]);
        double* lapl_weights = der->getLaplWeights(stencil[0]);

        for (int j = 0; j < stencil.size(); j++) {

            double grad_k_grad_phi_weight = gradK->x()*x_weights[j] + gradK->y()*y_weights[j] + gradK->z()*z_weights[j];
            double k_lapl_phi_weight = diffusivity*lapl_weights[j];

            // Our non-uniform diffusivity: grad(K) * grad(Phi) + K * lapl(Phi) = div(K*grad(Phi))
            // As K flattens (grad_k goes to 0) leaving only the scaled laplacian weights
            double weight = grad_k_grad_phi_weight + k_lapl_phi_weight;

            L(stencil[0],stencil[j]) = (FLOAT) weight;

            indx++;
        }
    }
    delete(gradK);
    return indx;
}


void NonUniformPoisson1_CL::fillInteriorRHS(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers)
{
    Vec3& v = centers[stencil[0]];
    double* x_weights = der->getXWeights(stencil[0]);
    double* y_weights = der->getYWeights(stencil[0]);
    double* z_weights = der->getZWeights(stencil[0]);
    double* lapl_weights = der->getLaplWeights(stencil[0]);
    int i = stencil[0];

    // Do I want the diffusivity (and its gradient) at all nodes or just the center?
    double diffusivity = exactSolution->diffuseCoefficient(v);
    Vec3* gradK = exactSolution->diffuseGradient(v);

    // Enable/disable this in the constructor/config file
    if (!use_discrete_rhs) {

        // exact laplacian
        if (use_uniform_diffusivity) {
            F[i] = (FLOAT)(exactSolution->laplacian(v));
        } else {
            double grad_k_grad_phi = exactSolution->xderiv(v)*gradK->x() + exactSolution->yderiv(v)*gradK->y() + exactSolution->zderiv(v)*gradK->z();
            double k_lapl_phi = exactSolution->laplacian(v) * diffusivity;
            F[i] = (FLOAT)(grad_k_grad_phi + k_lapl_phi);
        }
    } else {
        double exact = 0.;
        if (use_uniform_diffusivity) {
            double discrete_rhs = 0.;
            for (int j = 0; j < stencil.size(); j++) {
                Vec3& vj = centers[stencil[j]];
                discrete_rhs += lapl_weights[j] * exactSolution->at(vj);
            }
            // discrete laplacian
            F[i] = (FLOAT) (discrete_rhs);
            exact = exactSolution->laplacian(v);
        } else {
            double discrete_rhs = 0.;
            for (int j = 0; j < stencil.size(); j++) {
                Vec3& vj = centers[stencil[j]];
                double xderiv_approx = x_weights[j] * exactSolution->at(vj);
                double yderiv_approx = y_weights[j] * exactSolution->at(vj);
                double zderiv_approx = z_weights[j] * exactSolution->at(vj);
                double lapl_approx = lapl_weights[j] * exactSolution->at(vj);

                double grad_k_grad_phi_j = xderiv_approx*gradK->x() + yderiv_approx*gradK->y() + zderiv_approx*gradK->z();
                double k_lapl_phi_j = lapl_approx * diffusivity;

                discrete_rhs += grad_k_grad_phi_j + k_lapl_phi_j;
            }
            // discrete laplacian
            F[i] = (FLOAT) discrete_rhs;

            double grad_k_grad_phi = exactSolution->xderiv(v)*gradK->x() + exactSolution->yderiv(v)*gradK->y() + exactSolution->zderiv(v)
                                     *gradK->z();
            double k_lapl_phi = exactSolution->laplacian(v) * diffusivity;
            // Previously: double exact = exactSolution->laplacian(v);
            exact = grad_k_grad_phi + k_lapl_phi;
        }

        // Print error in discrete laplacian approximation:
        Vec3& v2 = centers[stencil[0]];
        double diff = fabs(exact - F[i]);
        double rel_diff = diff / fabs(exact);
        if (diff > 1e-9) {
            cout.precision(3);
            cout << "RHS[" << i << "] : Discrete laplacian differs by " << std::scientific << diff << " (rel: " << rel_diff << ", st.size: " << stencil.size() << ")" << endl;
        } else {
            cout << "RHS[" << i << "] : Good." << endl;
        }
    }
    delete(gradK);
}


//----------------------------------------------------------------------

template<typename T>
void NonUniformPoisson1_CL::write_to_file(boost::numeric::ublas::vector<T> vec, std::string filename)
{
    std::ofstream fout;
    fout.open(filename.c_str());
    for (int i = 0; i < vec.size(); i++) {
        fout << std::scientific << vec[i] << std::endl;
    }
    fout.close();
}

