#include <math.h>
#include "grid.h"
#include "ncar_poisson1.h"
#include "exact_solution.h"
#include <armadillo>

using namespace std;

NCARPoisson1::NCARPoisson1(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank) : exactSolution(_solution), rbf_centers(&subdomain_->G_centers), boundary_set(&subdomain_->global_boundary_nodes), der(der_), id(rank), subdomain(subdomain_) {
    nb_stencils = subdomain->Q_stencils.size();
    nb_rbf = subdomain->G_centers.size();

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
}

//----------------------------------------------------------------------

NCARPoisson1::~NCARPoisson1() {
}
//----------------------------------------------------------------------


// Advance the equation one time step using the GPU class to perform communication
// Depends on Constructor #2 to be used so that a GPU class exists within this class.

void NCARPoisson1::solve(Communicator* comm_unit) {

    exit(EXIT_FAILURE);
    if (subdomain == NULL) {
        cerr
                << "In " << __FILE__
                << " No GPU class passed to Constructor. Cannot perform intermediate communication/updates in solver."
                << endl;
        exit(EXIT_FAILURE);
    } else {

   
        vector<double>& s = sol[0];
        vector<double>& s1 = sol[1];

        // TODO: solve this in parallel
        //     comm_unit->broadcastObjectUpdates(subdomain);

        // Do NOT use GPU as buffer for computation
        // Only go up to the number of stencils since we solve for a subset of the values in U_G
        // Since U_G in R is at end of U_G vector we can ignore those.
        //for (int i = 0; i < s.size(); i++) {
        //    s[i] = subdomain->U_G[i];
        //}

        //der->computeDeriv(Derivative::LAPL, s, lapl_deriv);
        //der->computeDeriv(Derivative::LAPL, s, lapl_deriv);

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

        int nn = rbf_centers->size();
        int kk = boundary_set->size();

        // We are forming:
        //
        //   LA_i * a_i = F - LA_b * a_b
        //
        // where _i indicates interior and _b indicates boundary
        // LA_i is sparse and filled with laplacian weights from Derivative class

        // The derivative weights go into a matrix that is TotNumNodes x TotNumNodes
        // This is a sparse matrix though, so we're wasting memory and computation
        // TODO: replace this with a sparse solver
        arma::mat la_interior(nn,nn);
        la_interior.zeros();

        arma::mat la_boundary(nn, nn);
        la_boundary.zeros();
        
        arma::colvec f(nn);
     
        for (int i = 0; i < nn; i++) {
            f(i) = exactSolution->laplacian((*rbf_centers)[i],this->time);
        }

        arma::colvec a_b(nn);
        a_b.zeros();
        
        // Fill only the elements corresponding to boundary nodes
        for (int i = 0; i < boundary_set->size(); i++) {
            a_b((*boundary_set)[i]) = this->boundaryValues((*rbf_centers)[(*boundary_set)[i]]);
        }

        std::vector< std::vector<int> >& sten = subdomain->Q_stencils;

        // Fill Interior laplacian matrix
        for (int i = 0; i < sten.size(); i++) {
            arma::mat& weights = der->getLaplWeights(i);
            for (int j = 0; j < sten[i].size(); j++) {
                la_interior(i,sten[i][j]) = weights(j);
            }
        }

        // Fill Boundary laplacian matrix
        for (int i = 0; i < boundary_set->size(); i++) {
            int indx = (*boundary_set)[i];  // The ith boundary stencil
            arma::mat& weights = der->getLaplWeights(indx);
            for (int j = 0; j < sten[indx].size(); j++) {
                la_boundary(indx,sten[indx][j]) = weights(j);
                la_interior(indx,sten[indx][j]) = 0.;
            }
        }

        vector<double> sol_ex;
        vector<double> sol_error;

        sol_ex.resize(nb_stencils);
        sol_error.resize(nb_stencils);

        // exact solution
        for (int i = 0; i < nb_stencils; i++) {
            Vec3& v = (*rbf_centers)[i];
            sol_ex[i] = exactSolution->at(v, time);
            sol_error[i] = sol_ex[i] - s[i];
        }

        // print error to a file
        //	printf("nb_rbf= %d\n", nb_rbf);
        //	exit(0);

        char filename[256];
        sprintf(filename, "error.out.%d", id);

        FILE* fderr = fopen(filename, "w");
        for (int i = 0; i < nb_stencils; i++) {
            Vec3& v = (*rbf_centers)[i];
            fprintf(fderr, "%f %f %f %f\n", v[0], v[1], v[2], sol_error[i]);
        }
        fclose(fderr);

        sprintf(filename, "solution.out.%d", id);

        // print solution to a file
        FILE* fdsol = fopen(filename, "w");
        for (int i = 0; i < nb_stencils; i++) {
            Vec3& v = (*rbf_centers)[i];
            fprintf(fdsol, "%f %f %f %f\n", v.x(), v.y(), v.z(), s[i]);
        }
        fclose(fdsol);

        double nrm_ex = maxNorm(sol_ex);
        printf("exact max norm: %f\n", nrm_ex);
        double nrm_sol0 = maxNorm(s);
        printf("max norm(s[0]): %f\n", nrm_sol0);
        double nrm_error = maxNorm(sol_error);
        printf("nrm_error= %f\n", nrm_error);

        // And now we have full derivative calculated so we need to overwrite U_G
        for (int i = 0; i < s.size(); i++) {
            subdomain->U_G[i] = s[i];
        }
    }
    return;
}

//----------------------------------------------------------------------

void NCARPoisson1::initialConditions(std::vector<double> *solution) {
    vector<double>& s = sol[0];
    //double alpha = 1.00;
    //	vector<Vec3>* rbf_centers = getRbfCenters();

    //printf("%d, %d\n", s.size(), rbf_centers.size()); exit(0);

    for (int i = 0; i < s.size(); i++) {
        Vec3& v = (*rbf_centers)[i];
        //s[i] = exp(-alpha*v.square());
        //s[i] = 1. - v.square();
        s[i] = exactSolution->at(v, 0.);
        printf("filling: %f %f %f ==> %f\n", v.x(), v.y(), v.z(), s[i]);
        //s[i] = 1.0;
        //printf("s[%d]= %f\n", i, s[i]);
    }

    if (solution != NULL) {
        cout << "Filling solution parameter in addition to internal solution vector (initial condition)" << endl;
        for (int i = 0; i < s.size(); i++) {
            (*solution)[i] = s[i];
        }
    }
    //exit(0);

}

//----------------------------------------------------------------------

double NCARPoisson1::maxNorm() {
    double nrm = 0.;
    for (int i = 0; i < sol[0].size(); i++) {
        double s = abs(sol[0][i]);
        if (s > nrm)
            nrm = s;
    }

    //printf("max norm: %f\n", nrm);

    return nrm;
}
//----------------------------------------------------------------------

double NCARPoisson1::maxNorm(vector<double> sol) {
    double nrm = 0.;
    for (int i = 0; i < sol.size(); i++) {
        double s = abs(sol[i]);
        if (s > nrm)
            nrm = s;
    }

    //printf("max norm: %f\n", nrm);

    return nrm;
}
//----------------------------------------------------------------------


double NCARPoisson1::boundaryValues(Vec3& v) {
    return 0.;
}