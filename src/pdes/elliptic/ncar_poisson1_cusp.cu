#include <math.h>

#include <stdlib.h>
#include "grid.h"
#include "ncar_poisson1_cusp.h"
#include "exact_solution.h"

#include <cusp/hyb_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/blas.h>
#include <cusp/print.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/array1d.h>
#include <cusp/io/matrix_market.h>
#include <cusp/transpose.h>
using namespace std;

#define FLOAT float

NCARPoisson1_CUSP::NCARPoisson1_CUSP(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_) :
        NCARPoisson1(_solution, subdomain_, der_, rank, dim_num_)
{}

//----------------------------------------------------------------------

NCARPoisson1_CUSP::~NCARPoisson1_CUSP() {
}
//----------------------------------------------------------------------
// Solve the poisson system.
// NOTE: this routine is old and uses a possibly incorrect method for solving with the
// neumann boundary conditions. I am starting an alternate routine to solve the system
// in the same fashion that Joe solved the system.
void NCARPoisson1_CUSP::solve(Communicator* comm_unit) {

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
        cout << "Allocating GPU arrays " <<endl;

        //cusp::array1d<FLOAT, cusp::device_memory> exact(nn, 0);
        //cusp::array1d<FLOAT, cusp::device_memory> approx_sol(nn, 0);
       // cusp::array1d<FLOAT, cusp::device_memory> error(nn, 0);
       // cusp::array1d<FLOAT, cusp::device_memory> expected(nn, 0);
       // cusp::array1d<FLOAT, cusp::device_memory> diff_lapl(nn, 0);

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
new_eps = 1.;
            der->setEpsilon(new_eps);
            cout << "USING EPSILON: " << new_eps << endl;

            int numNonZeros = 0;
#if 1
            for (int i = 0; i < nb + ni; i++) {
                //subdomain->printStencil(subdomain->Q_stencils[i], "Q[i]");
                // Compute all derivatives for our centers and return the number of
                // weights that will be available
                numNonZeros += der->computeWeights(subdomain->G_centers, subdomain->Q_stencils[i], i, dim_num);
            }
#else
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
            cusp::coo_matrix<int, FLOAT, cusp::host_memory> L_host(nn, nn, numNonZeros);
            cusp::array1d<FLOAT, cusp::host_memory> F_host(nn, 0); // Initializes all elements to 0

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

#if 0
                // NEUMANN CONDITION:
                for (int j = 0; j < subdomain->Q_stencils[i].size(); j++) {
                        L_host.row_indices[indx] = i;
                        L_host.column_indices[indx] = subdomain->Q_stencils[i][j];
                        L_host.values[indx] = (center.x() / r) * x_weights[j] + (center.y()/r) * y_weights[j] + (center.z()/r) * z_weights[j];

                        // Remember to remove 1/r for the boundary condition: r d/dr(a/r) = 0
                        // When j == 0 we should have i = Q_stencil[i][j] (i.e., its the center element
                        if (j == 0) {
                            L_host.values[indx] -= 1./r;
                        }
                        indx++;
                 }
#else
#if 1
                // DIRICHLET CONDITION WITH 0s:
                for (int j = 0; j < subdomain->Q_stencils[i].size(); j++) {
                        L_host.row_indices[indx] = i;
                        L_host.column_indices[indx] = subdomain->Q_stencils[i][j];
                        L_host.values[indx] = 0;

                        if (j == 0) {
                            L_host.values[indx] = 1.f;
                        }
                        indx++;
                 }

#else
                // DIRICHLET CONDITION WITHOUT 0s
                        L_host.row_indices[indx] = i;
                        L_host.column_indices[indx] = i;
                        if (subdomain->Q_stencils[i][0] != i) {
                            cout << "WARNING!! i <> j" <<endl;
                            exit(EXIT_FAILURE);
                        }

                            L_host.values[indx] = 1.f;
                        indx++;
#endif
#endif
            }
//            cout << "INDX at end of boundary fill: " << indx << " NUM ROWS: " << nb+ni << endl;

            // Fill Interior weights
            for (int i = nb; i < nb+ni; i++) {
                double* lapl_weights = der->getLaplWeights(subdomain->Q_stencils[i][0]);
                for (int j = 0; j < subdomain->Q_stencils[i].size(); j++) {
                        L_host.row_indices[indx] = i;
                        L_host.column_indices[indx] = subdomain->Q_stencils[i][j];
                        L_host.values[indx] = (FLOAT)lapl_weights[j];
                        //cout << "lapl_weights[" << j << "] = " << lapl_weights[j] << endl;
                        indx++;
                 }
                Vec3& v = subdomain->G_centers[subdomain->Q_stencils[i][0]];
                F_host[i] = (FLOAT)exactSolution->laplacian(v.x(), v.y(), v.z(), 0.);
            }

            if ((indx - numNonZeros) != 0) {
                cerr << "WARNING! HOST MATRIX WAS NOT FILLED CORRECTLY. DISCREPANCY OF " << (indx - numNonZeros) << " NONZERO ELEMENTS!" << endl;
                exit(EXIT_FAILURE);
            }

           // cusp::print_matrix(F_host);

            // The way we fill the matrix is sorted by row so calling this has no effect
            //L_host.sort_by_row();
            cusp::io::write_matrix_market_file(L_host, "L.mtx");

            cusp::csr_matrix<int, FLOAT, cusp::device_memory> L_device;
            cusp::io::read_matrix_market_file(L_device, "L.mtx");

            cout << "READY TO SOLVE: " << endl;

            //cusp::csr_matrix<int, FLOAT, cusp::device_memory> L_device  = L_host;
            //cusp::transpose(L_host, L_device);

            cusp::array1d<FLOAT, cusp::device_memory> F_device = F_host;
            //cusp::print_matrix(F_host);

            cusp::array1d<FLOAT, cusp::device_memory> x_device(L_device.num_rows, 0.f);

            // set stopping criteria:
            //  iteration_limit    = 100
            //  relative_tolerance = 1e-6
            cusp::verbose_monitor<FLOAT> monitor(F_device, 100, 1e-6);

            // set preconditioner (identity)
            cusp::identity_operator<float, cusp::device_memory> M(L_device.num_rows, L_device.num_rows);

            // solve the linear system A * x = b with the BiConjugate Gradient Stabilized method
            cusp::krylov::bicgstab(L_device, x_device, F_device, monitor, M);

            // check residual norm
            cusp::array1d<float, cusp::device_memory> residual(L_device.num_rows, 0.0f);
            //L_device(x_device, residual);
            cusp::blas::axpby(x_device, F_device, residual, -1.0f, 1.0f);

            cout << "AXPBY RESIDUAL 2-NORM: " << cusp::blas::nrm2(residual) << endl;

            cusp::array1d<FLOAT, cusp::host_memory> x_host = x_device;
            cusp::array1d<FLOAT, cusp::host_memory> exact_H(F_device.size(), 0.f);

            //cout << "F = [";
            for (int i = 0; i < nb + ni; i++) {
                exact_H[subdomain->Q_stencils[i][0]] = (FLOAT)exactSolution->at(subdomain->G_centers[subdomain->Q_stencils[i][0]], 0.);
            //    cout << F_host[i] <<"; ";
            }
            //cout << "];" << endl;
            //cusp::array1d<FLOAT, cusp::device_memory> exact_D = exact_H;

            //cusp::array1d<FLOAT, cusp::device_memory> error_D(L_device.num_rows);
            // Compute our errors

            cout << "Error = [";
            cusp::array1d<FLOAT, cusp::host_memory> error_H(L_device.num_rows,0.f);
            for (int i = 0; i < L_device.num_rows; i++) {
                error_H[i] = x_host[i] - exact_H[i];
                cout << error_H[i] << ";\n";
            }
            cout << "]; " << endl;
            //error = (approx_sol - exact);
            //cusp::blas::axpby(exact_D, x_device, error_D, 1., -1.);

            //cusp::array1d<FLOAT, cusp::host_memory> error_H = error_D;

         //   cout << "Exact: "; cusp::print_matrix(exact_H);
         //   cout << "Sol: "; cusp::print_matrix(x_device);
         //   cout << "Error: "; cusp::print_matrix(error_H);
            //        expected = L*exact;
            //        diff_lapl = expected - F;

            prev_err_norm = err_norm;
            err_norm = cusp::blas::nrm2(error_H);

            cout << "2 Norm: " << err_norm << endl;

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

#if 0
        cusp::array1d<FLOAT, cusp::host_memory> results[5](nn);

        results[0] = approx_sol;
        results[1] = exact;
        results[2] = error;
        results.print("\n\nRESULTS\n (APPROX SOLUTION; \tEXACT SOLUTION; \tABS ERROR \tExpected Laplacian(Using L*ExactSolution)\t Diff EXPECTED & EXACT Laplacian\n");

        // results.col(3) = expected;
        // results.col(4) = diff_lapl;
        cusp::print_matrix(approx_sol);
#endif
        //A.print("Full Solution (A) = ");
        //exact.print("Exact = ");
        //error.print("Error = ");


        cout.flush();
    }
    return;
}

//----------------------------------------------------------------------
