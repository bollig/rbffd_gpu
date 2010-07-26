#include <math.h>

#include "grid.h"
#include "ncar_poisson1_cusp.h"
#include "exact_solution.h"

#include <cusp/hyb_matrix.h>
#include <cusp/print.h>
using namespace std;


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

        cusp::array1d<float, cusp::device_memory> exact(nn, 0);
        cusp::array1d<float, cusp::device_memory> approx_sol(nn, 0);
        cusp::array1d<float, cusp::device_memory> error(nn, 0);
        cusp::array1d<float, cusp::device_memory> expected(nn, 0);
        cusp::array1d<float, cusp::device_memory> diff_lapl(nn, 0);

        int iter = 0;
cout << "ENTERING EPSILON SEARCH LOOP " <<endl;
        while (err_norm > 1e-4 && iter < 10)
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

            der->setEpsilon(new_eps);
            cout << "USING EPSILON: " << new_eps << endl;

            for (int i = 0; i < nb + ni; i++) {
                //subdomain->printStencil(subdomain->Q_stencils[i], "Q[i]");
                // Compute all derivatives for our centers
                der->computeWeights(subdomain->G_centers, subdomain->Q_stencils[i], i, dim_num);
            }
#if 0
            // The derivative weights go into a matrix that is TotNumNodes x TotNumNodes
            // This is a sparse matrix though, so we're wasting memory and computation
            // TODO: replace this with a sparse solver
            arma::mat L(nn+1,nn+1);
            L.zeros();

            arma::colvec F(nn+1);
            F.zeros();

#if 1
            // This loop should add 1s to the far right column and bottom row; thereby removing the constant from the
            // possible solution and making the system nonsingular.
            for (int i = 0; i < nn; i++) {
                L(nn,i) = 1;
                L(i,nn) = 1;
            }
#else
            L(nn, nn) = 1;
#endif

            //    cout << "WARNING! using hardcoded constants for the boundaries!" << endl;
            //    cout << "WARNING! using x,y,z weights separately to compute d/dr!" << endl;
            // Block 1 (top left corner): d/dr weights for nb boundary points using nb+ni possible weights
            for (int i = 0; i < nb; i++) {
                //arma::mat& r_weights = der->getRWeights(subdomain->Q_stencils[i][0]);
                arma::mat& x_weights = der->getXWeights(subdomain->Q_stencils[i][0]);
                arma::mat& y_weights = der->getYWeights(subdomain->Q_stencils[i][0]);
                arma::mat& z_weights = der->getZWeights(subdomain->Q_stencils[i][0]);

                // DONT FORGET TO ADD IN THE -1/r on the stencil center weights
                Vec3& center = subdomain->G_centers[subdomain->Q_stencils[i][0]];
                double r = center.magnitude();
                //  r = 1.;
                for (int j = 0; j < subdomain->Q_stencils[i].size(); j++) {
                    //L(subdomain->Q_stencils[i][0],subdomain->Q_stencils[i][j]) = r_weights(j);        // Block 1 (weights for d/dr)
                    L(subdomain->Q_stencils[i][0],subdomain->Q_stencils[i][j]) = (center.x() / r) * x_weights(j) + (center.y()/r) * y_weights(j) + (center.z()/r) * z_weights(j);        // Block 1 (weights for d/dr)
                }

                if (r < 1e-8) {
                    cerr << "WARNING! VANISHING SPHERE RADIUS! CANNOT FILL -1/r in " << __FILE__ << endl;
                    exit(EXIT_FAILURE);
                }
                // Again, make sure we use Q_stencils[i][0] so we are forming the diagonals
                // correctly using the stencil center index (WARNING! this is not consistent for
                // domain decomposition... how to address this? TODO in the future..)
                L(subdomain->Q_stencils[i][0],subdomain->Q_stencils[i][0]) -= 1./r;
            }

            // Block 2 (bottom left corner): laplacian weights for ni interior points using nb+ni possible weights
            for (int i = 0; i < ni; i++) {
                int indx = i + nb; // offset into Q_stencils to get the interior stencils only
                arma::mat& l_weights = der->getLaplWeights(subdomain->Q_stencils[indx][0]);
                for (int j = 0; j < subdomain->Q_stencils[indx].size(); j++) {
                    L(subdomain->Q_stencils[indx][0],subdomain->Q_stencils[indx][j]) = l_weights(j);        // Block 1 (weights for laplacian)
                }
            }

            //L.print("L = ");

            for (int i = 0; i < ni; i ++) {
                int indx = i + nb;
                Vec3& v = subdomain->G_centers[subdomain->Q_stencils[indx][0]];
                F(subdomain->Q_stencils[indx][0]) = exactSolution->laplacian(v.x(), v.y(), v.z(), 0.);
            }
            F(nn) = 0.;

            //F.print("F = ");

            arma::mat sol = arma::solve(L,F);

            cout << "Measure sol(nn+1) = " << sol(nn) << endl;

            approx_sol = sol.rows(0,nn-1);

            // Get the subset of our full solution that corresponds to the solution we need
            // arma::mat A_sol = A.rows(0,nb+ni-1);

            // Fill our exact solution
            exact.zeros();
            for (int i = 0; i < nb + ni; i++) {
                exact(subdomain->Q_stencils[i][0]) = exactSolution->at(subdomain->G_centers[subdomain->Q_stencils[i][0]], 0.);
            }

            // Compute our errors
            error = (approx_sol - exact);

            //        expected = L*exact;
            //        diff_lapl = expected - F;

            prev_err_norm = err_norm;
            err_norm = this->maxNorm(error);
            cout << "INF NORM (ERROR) : " << err_norm << endl;

#endif
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
        cusp::array1d<float, cusp::host_memory> results[5](nn);

        results[0] = approx_sol;
        results[1] = exact;
        results[2] = error;
         results.print("\n\nRESULTS\n (APPROX SOLUTION; \tEXACT SOLUTION; \tABS ERROR \tExpected Laplacian(Using L*ExactSolution)\t Diff EXPECTED & EXACT Laplacian\n");
#endif
        // results.col(3) = expected;
        // results.col(4) = diff_lapl;
        cusp::print_matrix(approx_sol);

        //A.print("Full Solution (A) = ");
        //exact.print("Exact = ");
        //error.print("Error = ");


        cout.flush();
    }
    return;
}

//----------------------------------------------------------------------
