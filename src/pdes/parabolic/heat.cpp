#include <cmath>
#include <algorithm>

#include "timer_eb.h"
#include "rbffd/derivative.h"
#include "heat.h"
#include "exact_solutions/exact_solution.h"
#include "utils/norms.h"

using namespace std;
using namespace EB;

    Heat::Heat(ExactSolution* _solution, std::vector<NodeType>& rbf_centers_, int num_stencils, std::vector<size_t>& global_boundary_nodes_, Derivative* der_, int rank, double rel_err_max) 
:	rbf_centers(rbf_centers_), boundary_set(global_boundary_nodes_), der(der_),	id(rank), subdomain(NULL), exactSolution(_solution), rel_err_tol(rel_err_max) 
{
    nb_stencils = num_stencils;
    nb_rbf = rbf_centers.size();

    PI = acos(-1.);
    freq = PI / 2.;
    decay = 1.;

    time = 0.0; // physical time

    // solution + temporary array (for time advancement)
    sol[0].resize(nb_rbf);
    sol[1].resize(nb_rbf);


    // could resize inside the advancement function
    x_deriv.resize(nb_stencils);
    y_deriv.resize(nb_stencils);
    lapl_deriv.resize(nb_stencils);
    xx_deriv.resize(nb_stencils);
    yy_deriv.resize(nb_stencils);
    diffusion.resize(nb_stencils);
    diff_x.resize(nb_stencils);
    diff_y.resize(nb_stencils);

    setupTimers(); 
    // Cartesian-based Laplacian
    //grid.laplace();
}

    Heat::Heat(ExactSolution* _solution, Domain* subdomain_, Derivative* der_, int rank, double rel_err_max) 
: 	exactSolution(_solution), rbf_centers(subdomain_->getNodeList()), boundary_set(subdomain_->getBoundaryIndices()), der(der_), id(rank),subdomain(subdomain_), rel_err_tol(rel_err_max)
{
    nb_stencils = subdomain->getStencilsSize();
    nb_rbf = subdomain->getNodeListSize();

    PI = acos(-1.);
    freq = PI / 2.;
    decay = 1.;

    time = 0.0; // physical time

    // solution + temporary array (for time advancement)
    sol[0].resize(nb_rbf);
    sol[1].resize(nb_rbf);


    // could resize inside the advancement function
    x_deriv.resize(nb_stencils);
    y_deriv.resize(nb_stencils);
    lapl_deriv.resize(nb_stencils);
    xx_deriv.resize(nb_stencils);
    yy_deriv.resize(nb_stencils);
    diffusion.resize(nb_stencils);
    diff_x.resize(nb_stencils);
    diff_y.resize(nb_stencils);

    setupTimers(); 
    // Cartesian-based Laplacian
    //grid.laplace();
}
//----------------------------------------------------------------------
Heat::~Heat() {
}
//----------------------------------------------------------------------

void Heat::setupTimers() {
    tm["advance"] = new Timer("Heat::advanceOneStepWithComm (one step of the second order heat iteration)"); 
    tm["applyDerivsONLY"] = new Timer("[Heat] Apply derivatives to update solution");
}
#if 0
//----------------------------------------------------------------------
// explicit scheme: 
//  First order Euler (one step): 
//      f(t_{n}, u_{n}) = Laplacian(u_{n}) + ForcingTerm(t_{n}, u_{n})
//      u_{n+1} = u_{n} + dt * f(t_{n}, u_{n})
//
void Heat::advanceOneStep_FirstOrderEuler(Communicator* comm_unit) {
    tm["advance"]->start(); 
    if (subdomain == NULL) {
        cerr
            << "In HEAT.CPP: Wrong advanceOneStep* routine called! No Domain class passed to Constructor. Cannot perform intermediate communication/updates."
            << endl;
        exit(-10);
    } else {

        vector<double>& s = sol[0];
        vector<double>& s1 = sol[1];

        // Do NOT use Domain as buffer for computation
        // Only go up to the number of stencils since we solve for a subset of
        // the values in U_G
        // Since U_G in R is at end of U_G vector we can ignore those.
        for (int i = 0; i < s.size(); i++) {
            s[i] = subdomain->U_G[i];
        }

        // This is on the CPU or GPU depending on type of Derivative class used
        // (e.g., DerivativeCL will compute on GPU using OpenCL)
        der->computeDeriv(Derivative::LAPL, s, lapl_deriv);


        tm["applyDerivsONLY"]->start();  

        for (int i = 0; i < lapl_deriv.size(); i++) {
            Vec3& v = rbf_centers[i];
            double f = exactSolution->tderiv(v, time) - exactSolution->laplacian(v, time);
            // TODO: offload this to GPU. 
            s[i] = s[i] + dt* ( lapl_deriv[i] + f );
        }

        // reset boundary solution
        vector<size_t>& bnd_index = boundary_set; 
        int sz = bnd_sol.size();

        for (int i = 0; i < bnd_sol.size(); i++) {
            Vec3& v = rbf_centers[bnd_index[i]];
            s[bnd_index[i]] = bnd_sol[i];
        }
        tm["applyDerivsONLY"]->stop(); 

        comm_unit->broadcastObjectUpdates(subdomain);

        time += dt;
    }

}
#endif
//----------------------------------------------------------------------

// Advance the equation one time step using the Domain class to perform communication
// Depends on Constructor #2 to be used so that a Domain class exists within this class.
void Heat::advanceOneStepWithComm(Communicator* comm_unit) {
    tm["advance"]->start(); 
    if (subdomain == NULL) {
        cerr
            << "In HEAT.CPP: Wrong advanceOneStep* routine called! No Domain class passed to Constructor. Cannot perform intermediate communication/updates."
            << endl;
        exit(-10);
    } else {

        // This time advancement is second order.
        // It first updates s1 := s + 0.5 * dt * (lapl(s) + f)
        // Then it updates s := s + dt * (lapl(s1) + f)
        // However, in parallel our lapl(s) is size Q but depends on size Q+R
        // the lapl(s1) is size Q but depends on size Q+R; the key difference between
        // lapl(s) and lapl(s1) is that we have the set R for lapl(s) when we enter this
        // routine. Thus, we must call for the communicator comm_unit to update
        // the Domain object at the beginning/end and in the middle of this routine.
        // I think the best approach is to make the Heat class inherit from a ParallelPDE
        // type. ParallelPDE types will have pure virtual API for executing updates
        // on MPISendable types. Also I think ParallelPDE types should have internal
        // looping rather than require the main code control the loop. Mostly Im
        // thinking of the GLUT structure: glutInitFunc(), glutDisplayFunc(), ...
        // such that when glutMainLoop is called, a specific workflow is executed
        // on the registered callbacks (i.e. MPISendable update functions). This
        // would allow the main.cpp to initialize and forget, and allow more fine
        // grained control by registering custom routines.

        // compute laplace u^n
        // 2nd argument is vector<double>


        vector<double>& s = sol[0];
        vector<double>& s1 = sol[1];

        comm_unit->broadcastObjectUpdates(subdomain);
        // Do NOT use Domain as buffer for computation
        // Only go up to the number of stencils since we solve for a subset of the values in U_G
        // Since U_G in R is at end of U_G vector we can ignore those.
        for (int i = 0; i < s.size(); i++) {
            s[i] = subdomain->U_G[i];
        }

        // This is on the CPU or GPU depending on type of Derivative class used
        // (e.g., DerivativeCL will compute on GPU using OpenCL)
        der->computeDeriv(Derivative::LAPL, s, lapl_deriv);

        tm["applyDerivsONLY"]->start();  
        // Use 5 point Cartesian formula for the Laplacian
        //lapl_deriv = grid.computeCartLaplacian(s);
#if 0
        for (int i = 0; i < lapl_deriv.size(); i++) {
            Vec3& v = rbf_centers[i];
            printf("(local: %d), lapl(%f,%f,%f)= %f\tUpdated Solution=%f\n", i, v.x(), v.y(),v.z(),
                    lapl_deriv[i], s[i]);
        }
        //exit(0);
#endif

        // compute u^* = u^n + dt*lapl(u^n)

        printf("[advanceOneStepWithComm] heat, dt= %f, time= %f\n", dt, time);

        // second order time advancement if SECOND is defined
#define SECOND

        // explicit scheme
        for (int i = 0; i < lapl_deriv.size(); i++) {
            // first order
            Vec3& v = rbf_centers[i];
            //printf("dt= %f, time= %f\n", dt, time);
            double f = force(i, v, time*dt);
            //double f = 0.;
            //printf("force (local: %d) = %f\n", i, f);
            // TODO: offload this to GPU. 
            // s += alpha * lapl_deriv + f
            // NOTE: f is updated each timestep
            //       lapl_deriv is calculated on GPU (ideally by passing a GPU mem pointer into it) 
#ifdef SECOND
            s1[i] = s[i] + 0.5 * dt * (lapl_deriv[i] + f);
#else
            s[i] = s[i] + dt* ( lapl_deriv[i] + f);
#endif
        }

        // TODO: update to a different time stepping scheme
        time = time + dt;

        // reset boundary solution

        // assert bnd_sol.size() == bnd_index.size()
        vector<size_t>& bnd_index = boundary_set; //grid.getBoundary();
        int sz = bnd_sol.size();

        printf("nb bnd pts: %d\n", sz);

        for (int i = 0; i < bnd_sol.size(); i++) {
            // first order
            Vec3& v = rbf_centers[bnd_index[i]];
            //            printf("bnd[%d] = {%ld} %f, %f, %f\n", i, bnd_index[i], v.x(), v.y(), v.z());
#ifdef SECOND
            s1[bnd_index[i]] = bnd_sol[i];
#else
            s[bnd_index[i]] = bnd_sol[i];
#endif
        }
        tm["applyDerivsONLY"]->stop(); 

        // Now we need to make sure all CPUs have R2 (intermediate part of timestep)
        // Do NOT use Domain as buffer for computation
        for (int i = 0; i < s1.size(); i++) {
            subdomain->U_G[i] = s1[i];
        }
        comm_unit->broadcastObjectUpdates(subdomain);
        // Do NOT use Domain as buffer for computation
        for (int i = 0; i < s1.size(); i++) {
            s1[i] = subdomain->U_G[i];
            //  printf("s1[%d] = %f\n", i, s1[i]); 
        }

#ifdef SECOND
        // compute laplace u^*
        der->computeDeriv(Derivative::LAPL, s1, lapl_deriv);

        tm["applyDerivsONLY"]->start(); 
        // compute u^{n+1} = u^n + dt*lapl(u^*)
        for (int i = 0; i < lapl_deriv.size(); i++) {
            Vec3& v = rbf_centers[i];
            //printf("i= %d, nb_rbf= %d\n", i, nb_rbf);
            //v.print("v");
            double f = force(i, v, time+0.5*dt);
            // double f = 0.;
            s[i] = s[i] + dt * (lapl_deriv[i] + f); // RHS at time+0.5*dt
        }

        // reset boundary solution

        for (int i = 0; i < sz; i++) {
            s[bnd_index[i]] = bnd_sol[i];
        }
        tm["applyDerivsONLY"]->stop();
#endif

#if 0
        for (int i=0; i < s.size(); i++) {
            Vec3* v = rbf_centers[i];
            printf("(%f,%f): T(%d)=%f\n", v.x(), v.y(), i, s[i]);
        }
#endif

        // And now we have full derivative calculated so we need to overwrite U_G
        for (int i = 0; i < s.size(); i++) {
            subdomain->U_G[i] = s[i];
        }


        //comm_unit->broadcastObjectUpdates(subdomain);
        checkError(s, rbf_centers); 
    }
    tm["advance"]->end();
    return;
}

struct ltclass {
    bool operator() (size_t i, size_t j) { return (i<j); }
} srtobject; 


//----------------------------------------------------------------------
void Heat::checkError(std::vector<double>& sol_vec, std::vector<NodeType>& nodes, double rel_err_max)
{
    if (rel_err_max < 0) { 
        rel_err_max = rel_err_tol; 
    }

    vector<double> sol_error(sol_vec.size());
    vector<double> sol_exact(sol_vec.size());

    //std::cout << "======= TIME: " << time << " =======\n"; 
    for (int i = 0; i < sol_vec.size(); i++) {
        Vec3& v = nodes[i];
        sol_exact[i] = exactSolution->at(v, time);
        //  sol_error[i] = sol_exact[i] - sol_vec[i];
        //  printf("sol_error[%d] = %f\n", i, sol_error[i]); 
    }

    // Get a COPY of the indices because we want to sort them
    std::vector<size_t> bindices = subdomain->getBoundaryIndices(); 
    std::sort(bindices.begin(), bindices.end(), srtobject); 

    std::vector<double> sol_vec_bnd(bindices.size()); 
    std::vector<double> sol_exact_bnd(bindices.size()); 

    std::vector<double> sol_vec_int(subdomain->getNodeListSize() - bindices.size()); 
    std::vector<double> sol_exact_int(subdomain->getNodeListSize() - bindices.size()); 

    int i = 0;  // Index on boundary
    int k = 0;  // index on interior
    for (int j = 0; j < sol_vec.size(); j++) {
        // Skim off the boundary
        if (j == bindices[i]) {
            sol_vec_bnd[i] = sol_vec[j]; 
            sol_exact_bnd[i] = sol_exact[j]; 
            i++; 
            //  std::cout << "BOUNDARY: " << i << " / " << j << std::endl;
        } else {
            sol_vec_int[k] = sol_vec[j]; 
            sol_exact_int[k] = sol_exact[j]; 
            k++; 
            // std::cout << "INTERIOR: " << k << " / " << j <<  std::endl;
        }
    }

    //    writeErrorToFile(sol_error);

    calcSolNorms(sol_vec, sol_exact, "Full Domain", rel_err_max);  // Full domain
    calcSolNorms(sol_vec_int, sol_exact_int, "Interior", rel_err_max);  // Interior only
    calcSolNorms(sol_vec_bnd, sol_exact_bnd, "Boundary", rel_err_max);  // Boundary only
}

void Heat::calcSolNorms(std::vector<double>& sol_vec, std::vector<double>& sol_exact, std::string label, double rel_err_max) {
    // We want: || x_exact - x_approx ||_{1,2,inf} 
    // and  || x_exact - x_approx ||_{1,2,inf} / || x_exact ||_{1,2,inf}

    double l1fabs = l1norm(sol_vec, sol_exact); 
    double l1rel = (l1norm(sol_exact) > 1e-10) ? l1fabs/l1norm(sol_exact) : 0.;
    double l2fabs = l2norm(sol_vec, sol_exact); 
    double l2rel = (l2norm(sol_exact) > 1e-10) ? l2fabs/l2norm(sol_exact) : 0.;
    double lifabs = linfnorm(sol_vec, sol_exact); 
    double lirel = (linfnorm(sol_exact) > 1e-10) ? lifabs/linfnorm(sol_exact) : 0.;

    // Only print this when we're looking at the global norms
    if (!label.compare("")) {
        printf("========= Norms For Current Solution ========\n"); 
        printf("Absolute =>  || x_exact - x_approx ||_p                    ,  where p={1,2,inf}\n"); 
        printf("Relative =>  || x_exact - x_approx ||_p / || x_exact ||_p  ,  where p={1,2,inf}\n"); 
    }

    printf("%s l1 error : Absolute = %f, Relative = %f\n", label.c_str(), l1fabs, l1rel );
    printf("%s l2 error : Absolute = %f, Relative = %f\n", label.c_str(), l2fabs, l2rel );
    printf("%s linf error : Absolute = %f, Relative = %f\n", label.c_str(), lifabs, lirel);

    if (l1rel > rel_err_max) {
        printf("[Heat] Error! l1 relative error (=%f%%) is too high to continue. We require %f%% or less.\n", 100.*l1rel, 100.*rel_err_max); 
        exit(EXIT_FAILURE);
    }
    if (l2rel > rel_err_max) {
        printf("[Heat] Error! l2 relative error (=%f%%) is too high to continue. We require %f%% or less.\n", 100.*l2rel, 100.*rel_err_max); 
        exit(EXIT_FAILURE);
    }
    if (lirel > rel_err_max) {
        printf("[Heat] Error! linf relative error (=%f%%) is too high to continue. We require %f%% or less.\n", 100.*lirel, 100.*rel_err_max); 
        exit(EXIT_FAILURE);
    }


}

//----------------------------------------------------------------------
void Heat::writeErrorToFile(std::vector<double>& err) {
    char filename[256];
    sprintf(filename, "error.out.%d", id);

    FILE* fderr = fopen(filename, "w");
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        fprintf(fderr, "%f %f %f %f\n", v[0], v[1], v[2], err[i]);
    }
    fclose(fderr);
#if 0
    sprintf(filename, "solution.out.%d", id);

    // print solution to a file
    FILE* fdsol = fopen(filename, "w");
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        fprintf(fdsol, "%f %f %f %f\n", v.x(), v.y(), v.z(), s[i]);
    }
    fclose(fdsol);
#endif 
}




//----------------------------------------------------------------------
void Heat::advanceOneStep(std::vector<double>* updated_solution) {
    // This time advancement is second order. 
    // It first updates s1 := s + 0.5 * dt * (lapl(s) + f)
    // Then it updates s := s + dt * (lapl(s1) + f) 
    // However, in parallel our lapl(s) is size Q but depends on size Q+R
    // the lapl(s1) is size Q but depends on size Q+R; the key difference between 
    // lapl(s) and lapl(s1) is that we have the set R for lapl(s) when we enter this
    // routine. Thus, we must call for the communicator comm_unit to update
    // the Domain object at the beginning/end and in the middle of this routine. 
    // I think the best approach is to make the Heat class inherit from a ParallelPDE 
    // type. ParallelPDE types will have pure virtual API for executing updates
    // on MPISendable types. Also I think ParallelPDE types should have internal 
    // looping rather than require the main code control the loop. Mostly Im 
    // thinking of the GLUT structure: glutInitFunc(), glutDisplayFunc(), ... 
    // such that when glutMainLoop is called, a specific workflow is executed
    // on the registered callbacks (i.e. MPISendable update functions). This 
    // would allow the main.cpp to initialize and forget, and allow more fine
    // grained control by registering custom routines. 

    // compute laplace u^n
    // 2nd argument is vector<double> 
    vector<double>& s = sol[0];
    vector<double>& s1 = sol[1];

    //	vector<Vec3>* rbf_centers = grid.getRbfCenters();

    // compute laplace u^n 
    if (updated_solution != NULL) {
        for (int i = 0; i < (*updated_solution).size(); i++) {
            s[i] = (*updated_solution)[i];
        }
        cout << "USING updated_solution" << endl;
    }
    der->computeDeriv(Derivative::LAPL, s, lapl_deriv);

    // Use 5 point Cartesian formula for the Laplacian
    //lapl_deriv = grid.computeCartLaplacian(s);
#if 1
    for (int i = 0; i < lapl_deriv.size(); i++) {
        Vec3& v = rbf_centers[i];
        printf("(local: %d), lapl(%f,%f,%f)= %f\t UpdatedSol = %f\n", i, v.x(), v.y(), v.z(),
                lapl_deriv[i], s[i]);
    }
    //exit(0);
#endif

    // compute u^* = u^n + dt*lapl(u^n)

    printf("heat, dt= %f, time= %f\n", dt, time);

    // second order time advancement if SECOND is defined
    //#define SECOND

    // explicit scheme
    for (int i = 0; i < nb_stencils; i++) {
        // first order
        Vec3& v = rbf_centers[i];
        //printf("dt= %f, time= %f\n", dt, time);
        double f = force(i, v, time);
        //printf("force (local: %d) = %f\n", i, f);
        //printf("Before %f,", s[i]);
#ifdef SECOND
        s1[i] = s[i] + 0.5 * dt * (lapl_deriv[i] + f);
#else
        s[i] = s[i] + dt* ( lapl_deriv[i] + f);
#endif
        //printf("After %f\n", s[i]);
    }

    time = time + dt;

    // reset boundary solution

    // assert bnd_sol.size() == bnd_index.size()
    vector<size_t>& bnd_index = boundary_set; //grid.getBoundary();
    int sz = bnd_sol.size();

    printf("nb bnd pts: %d\n", sz);

    for (int i = 0; i < sz; i++) {
        // first order
        Vec3& v = rbf_centers[bnd_index[i]];
        printf("bnd[%d] = {%ld} %f, %f, %f\n", i, bnd_index[i], v.x(), v.y(), v.z());
#ifdef SECOND
        s1[bnd_index[i]] = bnd_sol[i];
#else
        s[bnd_index[i]] = bnd_sol[i];
#endif
    }

#ifdef SECOND
    // compute laplace u^* 
    der->computeDeriv(Derivative::LAPL, s1, lapl_deriv);
    //cerr << "SECOND ORDER TIME" << endl;
    // compute u^{n+1} = u^n + dt*lapl(u^*)
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        //printf("i= %d, nb_rbf= %d\n", i, nb_rbf);
        //v.print("v");
        double f = force(i, v, time+0.5*dt);
        s[i] = s[i] + dt * (lapl_deriv[i] + f); // RHS at time+0.5*dt
    }

    // reset boundary solution
    for (int i = 0; i < sz; i++) {
        s[bnd_index[i]] = bnd_sol[i];
    }
#endif

#if 0
    for (int i=0; i < nb_rbf; i++) {
        Vec3* v = rbf_centers[i];
        printf("(%f,%f,%f): T(%d)=%f\n", v.x(), v.y(), v.z(), i, s[i]);
    }
#endif

    // solution analysis

    vector<double> sol_ex;
    vector<double> sol_error;

    sol_ex.resize(nb_stencils);
    sol_error.resize(nb_stencils);

    // exact solution
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        sol_ex[i] = exactSolution->at(v, time);
        sol_error[i] = sol_ex[i] - s[i];
        printf("%d Force: %f\tLapl: %f\t Solution: %f\t Exact: %f \t Error: %f\n",i, force(i, v, time), lapl_deriv[i], s[i], sol_ex[i], sol_error[i]);
    }

    // print error to a file
    //	printf("nb_rbf= %d\n", nb_rbf);
    //	exit(0);

    char filename[256];
    sprintf(filename, "error.out.%d", id);

    FILE* fderr = fopen(filename, "w");
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        fprintf(fderr, "%f %f %f %f\n", v[0], v[1], v[2], sol_error[i]);
    }
    fclose(fderr);

    sprintf(filename, "solution.out.%d", id);

    // print solution to a file
    FILE* fdsol = fopen(filename, "w");
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        fprintf(fdsol, "%f %f %f %f\n", v.x(), v.y(), v.z(), s[i]);
    }
    fclose(fdsol);

    double nrm_ex = maxNorm(sol_ex);
    printf("(final simulation time: %f) max norm of exact solution: %f\n", time, nrm_ex);
    double nrm_sol0 = maxNorm(s);
    printf("(final simulation time: %f) max norm of computed solution: %f\n", time, nrm_sol0);
    double nrm_error = maxNorm(sol_error);
    printf("(final simulationtime: %f) max nrm of error= %f\n", time, nrm_error);

    if (updated_solution != NULL) {
        for (int i = 0; i < s.size(); i++) {
            (*updated_solution)[i] = s[i];
        }
    }

    return;

}
//----------------------------------------------------------------------
void Heat::advanceOneStepDivGrad() {
    // instead of computing the Laplacian, compute div(grad)

    // compute laplace u^n
    // 2nd argument is vector<double> 
    vector<double>& s = sol[0];
    vector<double>& s1 = sol[1];

    //	vector<Vec3>* rbf_centers = grid.getRbfCenters();

    // compute laplace u^n 
    der->computeDeriv(Derivative::X, s, x_deriv);
    der->computeDeriv(Derivative::Y, s, y_deriv);

#if 0
    for (int i=0; i < nb_rbf; i++) {
        printf("x,y deriv: %f, %f\n", x_deriv[i], y_deriv[i]);
    }
    double nxd = maxNorm(x_deriv);
    printf(" max norm of x_deriv = %f\n", nxd);
    double nyd = maxNorm(y_deriv);
    printf(" max norm of y_deriv = %f\n", nyd);
    exit(0);
#endif

    der->computeDeriv(Derivative::X, x_deriv, xx_deriv);
    der->computeDeriv(Derivative::Y, y_deriv, yy_deriv);
    //der.computeDeriv(Derivative::LAPL, s, lapl_deriv);

    for (int i = 0; i < s.size(); i++) {
        lapl_deriv[i] = xx_deriv[i] + yy_deriv[i];
    }

    // compute u^* = u^n + dt*lapl(u^n)

    printf("heat, dt= %f, time= %f\n", dt, time);

    // second order time advancement if SECOND is defined
#define SECOND

    // explicit scheme
    for (int i = 0; i < nb_stencils; i++) {
        // first order
        Vec3& v = rbf_centers[i];
        //printf("dt= %f, time= %f\n", dt, time);
        double f = exactSolution->tderiv(v, time) - exactSolution->laplacian(v, time);
        //printf("f= %f\n", f);
#ifdef SECOND
        s1[i] = s[i] + 0.5 * dt * (lapl_deriv[i] + f);
#else
        s[i] = s[i] + dt* ( lapl_deriv[i] + f);
#endif
    }

    time = time + dt;

    // reset boundary solution

    // assert bnd_sol.size() == bnd_index.size()
    vector<size_t>& bnd_index = boundary_set; //grid.getBoundary();
    int sz = bnd_sol.size();

    //printf("nb bnd pts: %d\n", sz);

    for (int i = 0; i < sz; i++) {
        // first order
        Vec3& v = rbf_centers[bnd_index[i]];
        //printf("bnd[%d] = %f, %f\n", i, v.x(), v.y());
#ifdef SECOND
        s1[bnd_index[i]] = bnd_sol[i];
#else
        s[bnd_index[i]] = bnd_sol[i];
#endif
    }

    double ns1 = maxNorm(s1);
    //printf(" max norm of s1 = %f\n", ns1);
    //exit(0);

#ifdef SECOND
    // compute laplace u^* 
    //der.computeDeriv(Derivative::LAPL, s1, lapl_deriv);
    // compute laplace u^n 
    der->computeDeriv(Derivative::X, s1, x_deriv);
    der->computeDeriv(Derivative::Y, s1, y_deriv);
    der->computeDeriv(Derivative::X, x_deriv, xx_deriv);
    der->computeDeriv(Derivative::Y, y_deriv, yy_deriv);

    for (int i = 0; i < s.size(); i++) {
        lapl_deriv[i] = xx_deriv[i] + yy_deriv[i];
    }

    // compute u^{n+1} = u^n + dt*lapl(u^*)
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        //printf("i= %d, nb_rbf= %d\n", i, nb_rbf);
        //v.print("v");
        double f = exactSolution->tderiv(v, time + 0.5 * dt) - exactSolution->laplacian(v, time + 0.5 * dt);
        s[i] = s[i] + dt * (lapl_deriv[i] + f); // RHS at time+0.5*dt
    }

    // reset boundary solution

    for (int i = 0; i < sz; i++) {
        s[bnd_index[i]] = bnd_sol[i];
    }
#endif

#if 0
    for (int i=0; i < nb_rbf; i++) {
        Vec3* v = rbf_centers[i];
        printf("(%f,%f): T(%d)=%f\n", v.x(), v.y(), i, s[i]);
    }
#endif

    // solution analysis

    vector<double> sol_ex;
    vector<double> sol_error;

    sol_ex.resize(nb_stencils);
    sol_error.resize(nb_stencils);

    // exact solution
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        sol_ex[i] = exactSolution->at(v, time);
        sol_error[i] = sol_ex[i] - s[i];
    }

    // print error to a file
    //printf("nb_rbf= %d\n", nb_rbf);
    //exit(0);

    FILE* fderr = fopen("error.out", "w");
    for (int i = 0; i < nb_rbf; i++) {
        Vec3& v = rbf_centers[i];
        fprintf(fderr, "%f %f %f %f\n", v[0], v[1], v[2], sol_error[i]);
    }
    fclose(fderr);

    // print solution to a file
    FILE* fdsol = fopen("solution.out", "w");
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        fprintf(fdsol, "%f %f %f %f\n", v.x(), v.y(), v.z(), s[i]);
    }
    fclose(fdsol);

    double nrm_ex = maxNorm(sol_ex);
    printf("(time: %f) exact max norm: %f\n", time, nrm_ex);
    double nrm_sol0 = maxNorm(s);
    printf("(time: %f) max norm(s[0]): %f\n", time, nrm_sol0);
    double nrm_error = maxNorm(sol_error);
    printf("(time: %f) nrm_error= %f\n", time, nrm_error);

    return;

#undef SECOND
}
//----------------------------------------------------------------------
void Heat::advanceOneStepTwoTerms() {
    // instead of computing the div(D grad)T, compute
    //     grad(D).grad(T) + D lapl(T)

    // compute laplace u^n
    // 2nd argument is vector<double> 
    vector<double>& s = sol[0];
    vector<double>& s1 = sol[1];

    //	vector<Vec3>* rbf_centers = grid.getRbfCenters();

    // compute laplace u^n 
    computeDiffusion(s);
    der->computeDeriv(Derivative::X, s, x_deriv);
    der->computeDeriv(Derivative::Y, s, y_deriv);
    der->computeDeriv(Derivative::X, diffusion, diff_x);
    der->computeDeriv(Derivative::Y, diffusion, diff_y);
    der->computeDeriv(Derivative::LAPL, s, lapl_deriv);

#if 0
    for (int i=0; i < nb_rbf; i++) {
        printf("x,y deriv: %f, %f\n", x_deriv[i], y_deriv[i]);
    }
    double nxd = maxNorm(x_deriv);
    printf(" max norm of x_deriv = %f\n", nxd);
    double nyd = maxNorm(y_deriv);
    printf(" max norm of y_deriv = %f\n", nyd);
    exit(0);
#endif

    //der.computeDeriv(Derivative::X, x_deriv, xx_deriv);
    //der.computeDeriv(Derivative::Y, y_deriv, yy_deriv);
    //der.computeDeriv(Derivative::LAPL, s, lapl_deriv);

    //for (int i=0; i < s.size(); i++) {
    //lapl_deriv[i] = xx_deriv[i] + yy_deriv[i];
    //}

    // compute u^* = u^n + dt*lapl(u^n)

    printf("heat, dt= %f, time= %f\n", dt, time);

    // second order time advancement if SECOND is defined
#define SECOND

    // explicit scheme
    for (int i = 0; i < nb_stencils; i++) {
        // first order
        Vec3& v = rbf_centers[i];
        //printf("dt= %f, time= %f\n", dt, time);
        double f = exactSolution->tderiv(v, time) - exactSolution->laplacian(v, time);
        //printf("f= %f\n", f);
        double grad = diff_x[i] * x_deriv[i] + diff_y[i] * y_deriv[i];
#ifdef SECOND
        s1[i] = s[i] + 0.5 * dt * (grad + diffusion[i] * lapl_deriv[i] + f);
#else
        s[i] = s[i] + dt * (grad + diffusion[i]*lapl_deriv[i] + f);
#endif
    }

    time = time + dt;

    // reset boundary solution

    // assert bnd_sol.size() == bnd_index.size()
    vector<size_t>& bnd_index = boundary_set; //grid.getBoundary();
    int sz = bnd_sol.size();

    //printf("nb bnd pts: %d\n", sz);

    for (int i = 0; i < sz; i++) {
        // first order
        Vec3& v = rbf_centers[bnd_index[i]];
        //printf("bnd[%d] = %f, %f\n", i, v.x(), v.y());
#ifdef SECOND
        s1[bnd_index[i]] = bnd_sol[i];
#else
        s[bnd_index[i]] = bnd_sol[i];
#endif
    }

    double ns1 = maxNorm(s1);
    //printf(" max norm of s1 = %f\n", ns1);
    //exit(0);

#ifdef SECOND
    // compute laplace u^* 
    //der.computeDeriv(Derivative::LAPL, s1, lapl_deriv);
    // compute laplace u^n 
    computeDiffusion(s1);
    der->computeDeriv(Derivative::X, s1, x_deriv);
    der->computeDeriv(Derivative::Y, s1, y_deriv);
    der->computeDeriv(Derivative::X, diffusion, diff_x);
    der->computeDeriv(Derivative::Y, diffusion, diff_y);
    der->computeDeriv(Derivative::LAPL, s, lapl_deriv);

    // compute u^{n+1} = u^n + dt*lapl(u^*)
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        //printf("i= %d, nb_rbf= %d\n", i, nb_rbf);
        //v.print("v");
        double f = exactSolution->tderiv(v, time + 0.5 * dt) - exactSolution->laplacian(v, time + 0.5 * dt);
        double grad = diff_x[i] * x_deriv[i] + diff_y[i] * y_deriv[i];
        s[i] = s[i] + dt * (grad + diffusion[i] * lapl_deriv[i] + f);
    }

    // reset boundary solution

    for (int i = 0; i < sz; i++) {
        s[bnd_index[i]] = bnd_sol[i];
    }
#endif

#if 0
    for (int i=0; i < nb_rbf; i++) {
        Vec3* v = rbf_centers[i];
        printf("(%f,%f): T(%d)=%f\n", v.x(), v.y(), i, s[i]);
    }
#endif

    // solution analysis

    vector<double> sol_ex;
    vector<double> sol_error;

    sol_ex.resize(nb_stencils);
    sol_error.resize(nb_stencils);

    // exact solution
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        sol_ex[i] = exactSolution->at(v, time);
        sol_error[i] = sol_ex[i] - s[i];
    }

    // print error to a file
    //printf("nb_rbf= %d\n", nb_rbf);
    //exit(0);

    FILE* fderr = fopen("error.out", "w");
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        fprintf(fderr, "%f %f %f %f\n", v[0], v[1], v[2], sol_error[i]);
    }
    fclose(fderr);

    // print solution to a file
    FILE* fdsol = fopen("solution.out", "w");
    for (int i = 0; i < nb_stencils; i++) {
        Vec3& v = rbf_centers[i];
        fprintf(fdsol, "%f %f %f %f\n", v.x(), v.y(), v.z(), s[i]);
    }
    fclose(fdsol);

    double nrm_ex = maxNorm(sol_ex);
    printf("(time: %f) exact max norm: %f\n", time, nrm_ex);
    double nrm_sol0 = maxNorm(s);
    printf("(time: %f) max norm(s[0]): %f\n", time, nrm_sol0);
    double nrm_error = maxNorm(sol_error);
    printf("(time: %f) nrm_error= %f\n", time, nrm_error);

    return;

#undef SECOND
}
//----------------------------------------------------------------------
void Heat::initialConditions(std::vector<double>* solution) {
    vector<double>& s = sol[0];
    //double alpha = 1.00;
    //	vector<Vec3>* rbf_centers = getRbfCenters();

    //printf("%d, %d\n", s.size(), rbf_centers.size()); exit(0);

    printf("=========== Initial Conditions ===========\n");

    printf("Using ExactSolution to fill initial conditions for interior\n");
    for (int i = 0; i < s.size(); i++) {
        Vec3& v = rbf_centers[i];
        //s[i] = exp(-alpha*v.square());
        //s[i] = 1. - v.square();
        s[i] = exactSolution->at(v, 0.);
        // printf("interior: %f %f %f ==> %f\n", v.x(), v.y(), v.z(),s[i]);
        //s[i] = 1.0;
        //printf("s[%d]= %f\n", i, s[i]);
    }
    //exit(0);

    vector<size_t>& bnd_index = boundary_set; //grid.getBoundary();
    int sz = bnd_index.size();
    bnd_sol.resize(sz);


    printf("Copying solution to bnd_sol (boundary solution buffer)\n"); 
    for (int i = 0; i < sz; i++) {
        bnd_sol[i] = s[bnd_index[i]];
        Vec3& v = rbf_centers[bnd_index[i]];
        // printf("boundary: %f %f %f ==> %f\n", v.x(), v.y(), v.z(),bnd_sol[i]);
    }

    if (solution != NULL) {
        cout << "Copying initial conditions to output buffer" << endl;
        for (int i = 0; i < s.size(); i++) {
            Vec3& v = rbf_centers[i];
            (*solution)[i] = s[i];
            // printf("interior: %f %f %f ==> %f\n", v.x(), v.y(), v.z(),s[i]);
        }
    }
    printf("============ End Initial Conditions ===========\n");
    //exit(0);
}
//----------------------------------------------------------------------
void Heat::computeDiffusion(vector<double>& sol) {
    //	vector<Vec3>* rbf_centers = getRbfCenters();

    for (int i = 0; i < sol.size(); i++) {
        Vec3& v = rbf_centers[i];
        diffusion[i] = sol[i];
    }
}
//----------------------------------------------------------------------
double Heat::maxNorm() {
    double nrm = 0.;
    for (int i = 0; i < sol[0].size(); i++) {
        double s = fabs(sol[0][i]);
        if (s > nrm)
            nrm = s;
    }

    //printf("max norm: %f\n", nrm);

    return nrm;
}
//----------------------------------------------------------------------
double Heat::maxNorm(vector<double>& sol) {
    double nrm = 0.;
    for (int i = 0; i < sol.size(); i++) {
        double s = fabs(sol[i]);
        if (s > nrm)
            nrm = s;
    }

    //printf("max norm: %f\n", nrm);

    return nrm;
}

//----------------------------------------------------------------------
// The forcing term.
// Here we get the term using the method of manufactured solutions.
// (plugin lapl(u) to get rhs of the equation). 
double Heat::force(size_t v_indx, Vec3& v, double t) {

    //cout << "TDERIV: " << exactSolution->tderiv(v,t) << "\tLapl: " << exactSolution->laplacian(v,t) << endl;
    //return exactSolution->laplacian(v,t);
#if 1
    // Gordon's suggestion
    return exactSolution->tderiv(v, t) - exactSolution->laplacian(v, t);
#else 
#if 0
//    return sol[0][v_indx] - exactSolution->at(v, t); 
    return exactSolution->at(v, t) - sol[0][v_indx]; 
#else 
    // Exact update we expect at the current time 
    // minus what we actually have at the current time tells us
    // exactly what our forcing needs to be
    return exactSolution->laplacian(v,t) - lapl_deriv[v_indx];
#endif 
#endif 
}
//----------------------------------------------------------------------
