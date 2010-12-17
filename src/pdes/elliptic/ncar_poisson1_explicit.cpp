#include <math.h>
#include "grid.h"
#include "ncar_poisson1_explicit.h"
#include "exact_solution.h"
#include <armadillo>

using namespace std;

NCARPoisson1Explicit::NCARPoisson1Explicit(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank) : exactSolution(_solution), rbf_centers(&subdomain_->G_centers), boundary_set(&subdomain_->global_boundary_nodes), der(der_), id(rank), subdomain(subdomain_) {
    nb_stencils = subdomain->Q_stencils.size();
    nb_rbf = subdomain->G_centers.size();

    time = 0.0; // physical time

    // solution + temporary array (for time advancement)
    sol.resize(nb_rbf);
    lapl_deriv.resize(nb_stencils);

    // could resize inside the advancement function
    x_deriv.resize(nb_stencils);
    y_deriv.resize(nb_stencils);
    xx_deriv.resize(nb_stencils);
    yy_deriv.resize(nb_stencils);
}

//----------------------------------------------------------------------

NCARPoisson1Explicit::~NCARPoisson1Explicit() {
}
//----------------------------------------------------------------------


// Advance the equation one time step using the GPU class to perform communication
// Depends on Constructor #2 to be used so that a GPU class exists within this class.

void NCARPoisson1Explicit::solve(Communicator* comm_unit) {

    if (subdomain == NULL) {
        cerr
                << "In " << __FILE__
                << " No GPU class passed to Constructor. Cannot perform intermediate communication/updates in solver."
                << endl;
        exit(EXIT_FAILURE);

    } else {

   printf("HERE!\n");
        // TODO: solve this in parallel
       // comm_unit->broadcastObjectUpdates(subdomain);

        // Do NOT use GPU as buffer for computation
        // Only go up to the number of stencils since we solve for a subset of the values in U_G
        // Since U_G in R is at end of U_G vector we can ignore those.
        for (int i = 0; i < sol.size(); i++) {
      //      sol[i] = subdomain->U_G[i];
    //        printf("Copying: %f\n", sol[i]);
        }

        for (int i = 0; i < subdomain->Q_stencils.size(); i++) {
            der->computeWeightsSVD(subdomain->G_centers, subdomain->Q_stencils[i], i, "lapl");
        }

        // The explicit laplacian will end up in lapl_deriv.
        // For an the basic poisson problem, der
        // has already combined weights and forcing terms (s)
        // to get the desired solution approximation.
        der->computeDeriv(Derivative::LAPL, sol, lapl_deriv);


        for (int i = 0; i < lapl_deriv.size(); i++) {
          Vec3& v = (*rbf_centers)[i];
          printf("(local: %d), lapl(%f,%f,%f)= %f\tExact: %f\n", i, v.x(), v.y(), v.z(),
                    lapl_deriv[i], exactSolution->at(v,time));
          // Update s because want the solution cached inside this object.
            sol[i] = lapl_deriv[i];
        }

        vector<double> sol_ex;
        vector<double> sol_error;

        sol_ex.resize(nb_stencils);
        sol_error.resize(nb_stencils);

        // exact solution
        for (int i = 0; i < nb_stencils; i++) {
            Vec3& v = (*rbf_centers)[i];
            sol_ex[i] = exactSolution->at(v, time);
            sol_error[i] = sol_ex[i] - sol[i];
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
            fprintf(fdsol, "%f %f %f %f\n", v.x(), v.y(), v.z(), sol[i]);
        }
        fclose(fdsol);

        double nrm_ex = maxNorm(sol_ex);
        printf("exact max norm: %f\n", nrm_ex);
        double nrm_sol0 = maxNorm(sol);
        printf("max norm(s[0]): %f\n", nrm_sol0);
        double nrm_error = maxNorm(sol_error);
        printf("nrm_error= %f\n", nrm_error);

        // And now we have full derivative calculated so we need to overwrite U_G
        for (int i = 0; i < sol.size(); i++) {
            subdomain->U_G[i] = sol[i];
        }
    }
    return;
}

//----------------------------------------------------------------------

void NCARPoisson1Explicit::initialConditions(std::vector<double> *solution) {
    vector<double>& s = sol;
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

double NCARPoisson1Explicit::maxNorm() {
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

double NCARPoisson1Explicit::maxNorm(vector<double> sol) {
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


double NCARPoisson1Explicit::boundaryValues(Vec3& v) {
    return 0.;
}
