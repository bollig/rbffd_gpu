//#define VIENNACL_HAVE_UBLAS 1
#include <stdlib.h>
#include <math.h>

#include "grid.h"
#include "ncar_poisson1_cl.h"
#include "exact_solution.h"

// The GPU/OpenCL side sparse arrays come from ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/linalg/bicgstab.hpp"

#include <iostream>
#include <vector>

// The CPU Side sparse arrays come from Boost uBlas.
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
using namespace std;

// Set single or double precision here.
typedef float FLOAT;

NCARPoisson1_CL::NCARPoisson1_CL(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_) :
        NCARPoisson1(_solution, subdomain_, der_, rank, dim_num_)
{}

//----------------------------------------------------------------------

NCARPoisson1_CL::~NCARPoisson1_CL() {
}
//----------------------------------------------------------------------
// Solve the poisson system.
// NOTE: this routine is old and uses a possibly incorrect method for solving with the
// neumann boundary conditions. I am starting an alternate routine to solve the system
// in the same fashion that Joe solved the system.
void NCARPoisson1_CL::solve(Communicator* comm_unit) {

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
        double err_norm = 100;
        double prev_err_norm = 100;

        double left_eps = 0.1;
        double right_eps = 30.;
        double new_eps = left_eps;

        double prev_eps = left_eps;

        bool goodDirection, wentRight;

        int iter = 0;
        cout << "ENTERING EPSILON SEARCH LOOP " <<endl;
        while (err_norm > 1e-4 && iter < 1)
        {
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

            new_eps = left_eps + abs(right_eps - left_eps)/2.;
new_eps = 8.;
            der->setEpsilon(new_eps);
            cout << "USING EPSILON: " << new_eps << endl;

            int numNonZeros = 0;
#if 1
            // Compute all weights including those for the boundary nodes
            for (int i = 0; i < nb + ni; i++) {
                //subdomain->printStencil(subdomain->Q_stencils[i], "Q[i]");
                // Compute all derivatives for our centers and return the number of
                // weights that will be available
                numNonZeros += der->computeWeights(subdomain->G_centers, subdomain->Q_stencils[i], i, dim_num);
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

            cout << "Weights computed" << endl;

            // 1) Fill the ublas matrix on the CPU
            boost::numeric::ublas::compressed_matrix<FLOAT> L_host(nn+1, nn+1);
            boost::numeric::ublas::vector<FLOAT> F_host(nn+1);

            F_host(nn) = 0.;

            int indx = 0;

            // Fill Boundary weights
            for (int i = 0; i < nb; i++) {
                double* x_weights = der->getXWeights(subdomain->Q_stencils[i][0]);
                double* y_weights = der->getYWeights(subdomain->Q_stencils[i][0]);
                double* z_weights = der->getZWeights(subdomain->Q_stencils[i][0]);

                // DONT FORGET TO ADD IN THE -1/r on the stencil center weights
                Vec3& center = subdomain->G_centers[subdomain->Q_stencils[i][0]];
                double r = center.magnitude();

                if (r < 1e-8) {
                    cerr << "WARNING! VANISHING SPHERE RADIUS! CANNOT FILL -1/r in " << __FILE__ << endl;
                    exit(EXIT_FAILURE);
                }

// if 0 : Dirichlet boundary condition
// if 1 : Neuman boundary condition
#if 1
                // NEUMANN CONDITION:
                for (int j = 0; j < subdomain->Q_stencils[i].size(); j++) {
                        //L_host.row_indices[indx] = i;
                        //L_host.column_indices[indx] = subdomain->Q_stencils[i][j];
                        //L_host.values[indx] = (center.x() / r) * x_weights[j] + (center.y()/r) * y_weights[j] + (center.z()/r) * z_weights[j];
                        FLOAT value = (FLOAT)((center.x() * x_weights[j] + center.y() * y_weights[j]));// + (center.z()/r) * z_weights[j]);
                        // Remember to remove 1/r for the boundary condition: r d/dr(a/r) = 0
                        // When j == 0 we should have i = Q_stencil[i][j] (i.e., its the center element
#if 0
                        if (j == 0) {
                            //L_host.values[indx] -= 1./r;
                            value -= (FLOAT)(1./r);
                        }
#endif
                        L_host(subdomain->Q_stencils[i][0],subdomain->Q_stencils[i][j]) = value;
                        indx++;
                 }

#else
                // DIRICHLET CONDITION WITH 0s:
                for (int j = 0; j < subdomain->Q_stencils[i].size(); j++) {
                    if (j == 0) {
                        //L_host.values[indx] = 1.f;
                         L_host(subdomain->Q_stencils[i][0],subdomain->Q_stencils[i][j]) = (FLOAT)1.;
                       // L_host.insert_element(subdomain->Q_stencils[i][0],subdomain->Q_stencils[i][j], (FLOAT)1.);
                    } else {
                        //   L_host.row_indices[indx] = i;
                        //   L_host.column_indices[indx] = subdomain->Q_stencils[i][j];
                        //   L_host.values[indx] = 0;
                        L_host.insert_element(subdomain->Q_stencils[i][0],subdomain->Q_stencils[i][j], (FLOAT)0.);
                    }
                        indx++;
                 }
#endif
                F_host(i) = (FLOAT)0.;
            }


            for (int i = 0; i < nb+ni; i++) {
                // Constrain the solution to the correct Z
                L_host(subdomain->Q_stencils[i][0],nn) = 1;
                L_host(nn,subdomain->Q_stencils[i][0]) = 1;
                // Final constraint to make the Neumann boundary condition complete
                F_host(nn) += (FLOAT) (exactSolution->at(subdomain->G_centers[subdomain->Q_stencils[i][0]], 0.));
            }

            // Fill Interior weights
            for (int i = nb; i < nb+ni; i++) {
                double* lapl_weights = der->getLaplWeights(subdomain->Q_stencils[i][0]);
                for (int j = 0; j < subdomain->Q_stencils[i].size(); j++) {
                       // L_host.row_indices[indx] = i;
                       // L_host.column_indices[indx] = subdomain->Q_stencils[i][j];
                       // L_host.values[indx] = (float)lapl_weights[j];
                     L_host(subdomain->Q_stencils[i][0],subdomain->Q_stencils[i][j]) = (FLOAT) lapl_weights[j];
                        //cout << "lapl_weights[" << j << "] = " << lapl_weights[j] << endl;
                        indx++;
                 }

                Vec3& v = subdomain->G_centers[subdomain->Q_stencils[i][0]];
                // 0: RHS is discrete laplacian; 1: RHS exact laplacian
#if 0
                F_host[i] = (FLOAT)exactSolution->laplacian(v.x(), v.y(), v.z(), 0.);
#else
                F_host[i] = (FLOAT)0.;
                for (int j = 0; j < subdomain->Q_stencils[i].size(); j++) {
                    Vec3& vj = subdomain->G_centers[subdomain->Q_stencils[i][j]];
                    F_host[i] += (FLOAT)(lapl_weights[j] * exactSolution->laplacian(vj.x(), vj.y(), vj.z(), 0.));
                }
                Vec3& v2 = subdomain->G_centers[subdomain->Q_stencils[i][0]];
                cout << "EXACT_LAPLACIAN: " << exactSolution->laplacian(v.x(), v.y(), v.z(), 0.)
                     << "\t APPROX_LAPLACIAN: " << F_host[i] << endl;
#endif
            }

            if ((indx - numNonZeros) != 0) {
                cerr << "WARNING! HOST MATRIX WAS NOT FILLED CORRECTLY. DISCREPANCY OF " << (indx - numNonZeros) << " NONZERO ELEMENTS!" << endl;
                exit(EXIT_FAILURE);
            }


            cout << "Implicit system assembled" << endl;
            // cout << "L_HOST: " << L_host.size1() << "\t" << L_host.size2() << "\t" << L_host.filled1() << "\t" << L_host.filled2() << endl;
            // cout << L_host << endl;
            // cout << "F: " << F_host << endl;


            // 2) Convert to OpenCL space:

            viennacl::compressed_matrix<FLOAT, 1 /*Alignment(e.g.: 1,4,8)*/ > L_device(L_host.size1(), L_host.size2());
            viennacl::vector<FLOAT> F_device(F_host.size());
            viennacl::vector<FLOAT> x_device(F_host.size());

            boost::numeric::ublas::vector<FLOAT> x_host(F_host.size());

            cout << "Before copy to GPU" << endl;
            // copy to GPU
            copy(L_host, L_device);
            copy(F_host, F_device);

            //x_device = viennacl::linalg::prod(L_device, F_device);
            x_device = viennacl::linalg::solve(L_device, F_device, viennacl::linalg::bicgstab_tag(1.e-10, 600));
           // x_host = viennacl::linalg::solve(L_host, F_host, viennacl::linalg::gmres_tag());
           // x_host = viennacl::linalg::solve(L_host, F_host, viennacl::linalg::gmres_tag());
            viennacl::ocl::finish();

            cout << "Done with solve" << endl;

            // Copy solution to host
            copy(x_device, x_host);

            cout << "Results copied to host" << endl;

            boost::numeric::ublas::vector<FLOAT> exact_host(F_host.size());
            boost::numeric::ublas::vector<FLOAT> error_host(F_host.size());

            for (int i = 0; i < nb + ni; i++) {
                exact_host(subdomain->Q_stencils[i][0]) = (FLOAT) (exactSolution->at(subdomain->G_centers[subdomain->Q_stencils[i][0]], 0.));
            }

            error_host = exact_host - x_host;

            cout << "Writing results to disk" << endl;

            ofstream fout;
            fout.open("L.mtx");
            fout << L_host << endl;
            fout.close();
            fout.open("F.mtx");
            fout << F_host << endl;
            fout.close();
            fout.open("X_exact.mtx");
            fout << exact_host << endl;
            fout.close();
            fout.open("X_approx.mtx");
            fout << x_host << endl;
            fout.close();
            fout.open("R.mtx");
            fout << prod(L_host, exact_host) - F_host << endl;
            fout.close();
            fout.open("E_absolute.mtx");
            fout << error_host << endl;

            std::cout << "Relative residual || x_exact - x_approx ||_2 / || x_exact ||_2  = " << norm_2(exact_host - x_host) / norm_2(exact_host) << std::endl;
            std::cout << "Relative residual || A*x_exact - F ||_2 / || F ||_2  = " << norm_2(prod(L_host, exact_host) - F_host) / norm_2(F_host) << std::endl;
            std::cout << "Relative residual || A*x_approx - F ||_2 / || F ||_2  = " << norm_2(prod(L_host, x_host) - F_host) / norm_2(F_host) << std::endl;


            //cout << "Solution: " << x_host << endl;
           // cout << "Error: " << error_host << endl;


            prev_err_norm = err_norm;
            //err_norm = CL::blas::nrm2(error_H);

            //cout << "2 Norm: " << err_norm << endl;


            if (prev_err_norm > err_norm) {
                goodDirection = true;
            } else {
                goodDirection = false;
            }

            if (goodDirection) {
                if (wentRight) {
                    prev_eps = left_eps;
                    left_eps = new_eps;
                    wentRight = true;
                } else {
                    prev_eps = right_eps;
                    right_eps = new_eps;
                    wentRight = false;
                }
            } else {
                if (wentRight) {
                    left_eps = prev_eps;
                    right_eps = new_eps;
                    wentRight = false;
                } else {
                    right_eps = prev_eps;
                    left_eps = new_eps;
                    wentRight = true;
                }
            }

            iter ++;
        }

        cout.flush();
    }
    return;
}

//----------------------------------------------------------------------
