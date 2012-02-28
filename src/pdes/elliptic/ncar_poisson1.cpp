#include <stdlib.h>
#include <math.h>
#include "ncar_poisson1.h"
#include "exact_solutions/exact_solution.h"
#include <armadillo>
#include "rbffd/rbffd.h"

using namespace std;

NCARPoisson1::NCARPoisson1(ExactSolution* _solution, Grid* subdomain_, RBFFD* der_, int rank, int dim_num_, bool weightsAlreadyComputed) :
    dim_num(dim_num_), exactSolution(_solution), rbf_centers(&subdomain_->getNodeList()),
    boundary_set(&subdomain_->getBoundaryIndices()), der(der_), id(rank), subdomain(subdomain_),
    weightsPrecomputed(weightsAlreadyComputed),
    t1("[ncar_poisson_t1] Total"),
    t2("[ncar_poisson_t2] Compute Weights"),
    t3("[ncar_poisson_t3] Implicit Assemble"),
    t4("[ncar_poisson_t4] Solve"),
    t5("[ncar_poisson_t5] Solve w/o memcpy")
{
    nb_stencils = subdomain->getStencils().size();
    nb_rbf = subdomain->getNodeList().size();

    time = 0.0; // physical time

    // solution + temporary array (for time advancement)
    sol[0].resize(nb_rbf);
    sol[1].resize(nb_rbf);
    lapl_deriv.resize(nb_stencils);

    // could resize inside the advancement function
    x_deriv.resize(nb_stencils);
    y_deriv.resize(nb_stencils);
    xx_deriv.resize(nb_stencils);
    yy_deriv.resize(nb_stencils);

    // Do not run the L + L^T test
    boundary_condition = 0; // DIRICHLET
    disable_sol_constraint = false;
    use_discrete_rhs = true;
    use_uniform_diffusivity = false;
    test_dirichlet_lockdown = false;
}

#if 0
NCARPoisson1::NCARPoisson1(ProjectSettings* settings, ExactSolution* _solution, Domain* subdomain_, Derivative* der_, int rank, int dim_num_) :
    dim_num(dim_num_), exactSolution(_solution), rbf_centers(&subdomain_->getNodeList()),
    boundary_set(&subdomain_->getBoundaryIndices()), der(der_), id(rank), subdomain(subdomain_),
    t1(tm, "[ncar_poisson_t1] Total"),
    t2(tm, "[ncar_poisson_t2] Compute Weights"),
    t3(tm, "[ncar_poisson_t3] Implicit Assemble"),
    t4(tm, "[ncar_poisson_t4] Solve"),
    t5(tm, "[ncar_poisson_t5] Solve w/o memcpy")
{
    nb_stencils = subdomain->getStencils().size();
    nb_rbf = subdomain->getNodeList().size();

    time = 0.0; // physical time

    // solution + temporary array (for time advancement)
    sol[0].resize(nb_rbf);
    sol[1].resize(nb_rbf);
    lapl_deriv.resize(nb_stencils);

    // could resize inside the advancement function
    x_deriv.resize(nb_stencils);
    y_deriv.resize(nb_stencils);
    xx_deriv.resize(nb_stencils);
    yy_deriv.resize(nb_stencils);

    boundary_condition = settings->GetSettingAs<int>("BOUNDARY_CONDITION", ProjectSettings::required);
    disable_sol_constraint = settings->GetSettingAs<int>("DISABLE_SOL_CONSTRAINT", ProjectSettings::optional);
    use_discrete_rhs = settings->GetSettingAs<int>("USE_DISCRETE_RHS", ProjectSettings::required);
    use_uniform_diffusivity = settings->GetSettingAs<int>("USE_UNIFORM_DIFFUSIVITY", ProjectSettings::required);
    test_dirichlet_lockdown = settings->GetSettingAs<int>("TEST_DIRICHLET_LOCKDOWN", ProjectSettings::optional);
}
#endif 

//----------------------------------------------------------------------

NCARPoisson1::~NCARPoisson1() {
    t1.printAll();
}
//----------------------------------------------------------------------
// Solve the poisson system.
// NOTE: this routine is old and uses a possibly incorrect method for solving with the
// neumann boundary conditions. I am starting an alternate routine to solve the system
// in the same fashion that Joe solved the system.
void NCARPoisson1::solve(Communicator* comm_unit) {

    if (subdomain == NULL) {
        cerr
            << "In " << __FILE__
            << " No Domain class passed to Constructor. Cannot perform intermediate communication/updates in solver."
            << endl;
        exit(EXIT_FAILURE);
    } else {

        int nb = subdomain->getBoundaryIndices().size();
        // All interior and boundary nodes are included in the stencils.
        // The first nb getStencils() should be the global boundary nodes
        int ni = subdomain->getStencils().size() - nb;

        int nn = (nb + ni) ;
        double err_norm = 100;
        double prev_err_norm = 100;

        double left_eps = 0.1;
        double right_eps = 30.;
        double new_eps = left_eps;

        double prev_eps = left_eps;

        bool goodDirection = true; 
        bool wentRight = false;

        arma::colvec exact(nn);
        arma::colvec approx_sol(nn);
        arma::colvec error(nn);
        arma::colvec expected(nn);
        arma::colvec diff_lapl(nn);

        int iter = 0;

        while (err_norm > 1e-4 && iter < 10)
        {
            // TODO: solve this in parallel
            //     comm_unit->broadcastObjectUpdates(subdomain);

            // Do NOT use Domain as buffer for computation
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

            new_eps = left_eps + fabs(right_eps - left_eps)/2.;
            der->setEpsilon(new_eps);

            cout << "USING EPSILON: " << new_eps << "\t";

            if (!weightsPrecomputed) {
                der->computeAllWeightsForAllStencils(); 
            }

            // The derivative weights go into a matrix that is TotNumNodes x TotNumNodes
            // This is a sparse matrix though, so we're wasting memory and computation
            // TODO: replace this with a sparse solver
            arma::mat L(nn+1,nn+1);
            L.zeros();

            arma::colvec F(nn+1);
            F.zeros();


#if 0
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
                //arma::mat& r_weights = der->getRWeights(subdomain->getStencils()[i][0]);
                double* x_weights = der->getXWeights(subdomain->getStencils()[i][0]);
                double* y_weights = der->getYWeights(subdomain->getStencils()[i][0]);
                double* z_weights = der->getZWeights(subdomain->getStencils()[i][0]);

                // DONT FORGET TO ADD IN THE -1/r on the stencil center weights
                Vec3& center = subdomain->getNode(subdomain->getStencils()[i][0]);
                double r = center.magnitude();
                //  r = 1.;
                for (size_t j = 0; j < subdomain->getStencils()[i].size(); j++) {
                    //L(subdomain->getStencils()[i][0],subdomain->getStencils()[i][j]) = r_weights(j);        // Block 1 (weights for d/dr)
                    L(subdomain->getStencils()[i][0],subdomain->getStencils()[i][j]) = (center.x() / r) * x_weights[j] + (center.y()/r) * y_weights[j] + (center.z()/r) * z_weights[j];        // Block 1 (weights for d/dr)
                }

                if (r < 1e-8) {
                    cerr << "WARNING! VANISHING SPHERE RADIUS! CANNOT FILL -1/r in " << __FILE__ << endl;
                    exit(EXIT_FAILURE);
                }
                // Again, make sure we use getStencils()[i][0] so we are forming the diagonals
                // correctly using the stencil center index (WARNING! this is not consistent for
                // domain decomposition... how to address this? TODO in the future..)
                L(subdomain->getStencils()[i][0],subdomain->getStencils()[i][0]) -= 1./r;
            }

            // Block 2 (bottom left corner): laplacian weights for ni interior points using nb+ni possible weights
            for (int i = 0; i < ni; i++) {
                int indx = i + nb; // offset into getStencils() to get the interior stencils only
                double* l_weights = der->getLaplWeights(subdomain->getStencils()[indx][0]);
                for (size_t j = 0; j < subdomain->getStencils()[indx].size(); j++) {
                    L(subdomain->getStencils()[indx][0],subdomain->getStencils()[indx][j]) = l_weights[j];        // Block 1 (weights for laplacian)
                }
            }

            //L.print("L = ");

            for (int i = 0; i < ni; i ++) {
                int indx = i + nb;
                Vec3& v = subdomain->getNode(subdomain->getStencils()[indx][0]);
                F(subdomain->getStencils()[indx][0]) = exactSolution->laplacian(v.x(), v.y(), v.z(), 0.);
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
                exact(subdomain->getStencils()[i][0]) = exactSolution->at(subdomain->getNode(subdomain->getStencils()[i][0]), 0.);
            }

            // Compute our errors
            error = (approx_sol - exact);

            //        expected = L*exact;
            //        diff_lapl = expected - F;

            prev_err_norm = err_norm;
            err_norm = this->maxNorm(error.memptr(), error.n_rows, error.n_cols);
            cout << "INF NORM (ERROR) : " << err_norm << endl;


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

        arma::mat results(nn, 5);

        results.col(0) = approx_sol;
        results.col(1) = exact;
        results.col(2) = error;
        // results.col(3) = expected;
        // results.col(4) = diff_lapl;
        results.print("\n\nRESULTS\n (APPROX SOLUTION; \tEXACT SOLUTION; \tABS ERROR \tExpected Laplacian(Using L*ExactSolution)\t Diff EXPECTED & EXACT Laplacian\n");
        //A.print("Full Solution (A) = ");
        //exact.print("Exact = ");
        //error.print("Error = ");


        cout.flush();
    }
    return;
}

//----------------------------------------------------------------------



void NCARPoisson1::initialConditions(std::vector<double> *solution) {
    vector<double>& s = sol[0];
    //double alpha = 1.00;
    //	vector<Vec3>* rbf_centers = getRbfCenters();

    //printf("%d, %d\n", s.size(), rbf_centers.size()); exit(0);

    for (size_t i = 0; i < s.size(); i++) {
        Vec3& v = (*rbf_centers)[i];
        //s[i] = exp(-alpha*v.square());
        //s[i] = 1. - v.square();
        s[i] = exactSolution->at(v, 0.);
        //printf("filling: %f %f %f ==> %f\n", v.x(), v.y(), v.z(), s[i]);
        //s[i] = 1.0;
        //printf("s[%d]= %f\n", i, s[i]);
    }

    if (solution != NULL) {
        //cout << "Filling solution parameter in addition to internal solution vector (initial condition)" << endl;
        for (size_t i = 0; i < s.size(); i++) {
            (*solution)[i] = s[i];
        }
    }
    //exit(0);

}

//----------------------------------------------------------------------

double NCARPoisson1::maxNorm() {
    double nrm = 0.;
    for (size_t i = 0; i < sol[0].size(); i++) {
        double s = fabs(sol[0][i]);
        if (s > nrm)
            nrm = s;
    }

    //printf("max norm: %f\n", nrm);

    return nrm;
}
//----------------------------------------------------------------------

double NCARPoisson1::maxNorm(vector<double> sol) {
    double nrm = 0.;
    for (size_t i = 0; i < sol.size(); i++) {
        double s = fabs(sol[i]);
        if (s > nrm)
            nrm = s;
    }

    //printf("max norm: %f\n", nrm);

    return nrm;
}
//----------------------------------------------------------------------

double NCARPoisson1::maxNorm(double* sol, int nrows, int ncols) {
    double nrm = 0.;
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            double s = fabs(sol[i*ncols + j]);
            if (s > nrm)
                nrm = s;
        }
    }

    //printf("max norm: %f\n", nrm);
    return nrm;
}
//----------------------------------------------------------------------

double NCARPoisson1::boundaryValues(Vec3& v) {
    return 0.;
}
