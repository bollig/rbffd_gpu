#include "time_dependent_pde.h"

#include <iostream>

void TimeDependentPDE::fillInitialConditions(ExactSolution* exactSolution) {
    vector<SolutionType>& s = U_G;

    std::set<int>& Q = grid_ref.Q;			// All stencil centers in this CPU's QUEUE							

    // Only fill the solution values under this procs control 
    // NOTE: any remaining values will be 0 initially

    //  printf("=========== Initial Conditions ===========\n");

    std::set<int>::iterator Q_iter; 

    // If we dont provide an exact solution, we'll default to 0's
    if (!exactSolution) {
        for (Q_iter = Q.begin(); Q_iter != Q.end(); Q_iter++) {
            NodeType& v = grid_ref.getNode(*Q_iter); 
            //s[grid_ref.g2l(*Q_iter)] = *Q_iter; 
            s[grid_ref.g2l(*Q_iter)] = 0; 
        }
    } else {
        for (Q_iter = Q.begin(); Q_iter != Q.end(); Q_iter++) {
            NodeType& v = grid_ref.getNode(grid_ref.g2l(*Q_iter)); 
            //            std::cout << "NODE ( " << grid_ref.g2l(*Q_iter) << " ) = " << v << std::endl;
            // evaluate the exact solution at the node at time 0.
            s[grid_ref.g2l(*Q_iter)] = exactSolution->at(v, 0.); 
        }
    }
    //    printf("============ End Initial Conditions ===========\n");
}

// Advancing requires: 
//  - computing an update to the current solution (i.e., calling applyWeightsForDerivs(currentSolution)) 
//  - applying the updates to the current solution (i.e., RK45 weighted summation of intermediate updates).
void TimeDependentPDE::advance(TimeScheme which, double delta_t) {
    switch (which) 
    {
        case FIRST_EULER: 
            advanceFirstEuler(delta_t); 
            break; 
        case SECOND_EULER: 
            advanceSecondEuler(delta_t);
            break;  
        default: 
            std::cout << "[TimeDependentPDE] Invalid TimeScheme specified. Bailing...\n";
            exit(EXIT_FAILURE); 
            break; 
    };
    cur_time += delta_t; 
}

void TimeDependentPDE::advanceFirstEuler(double dt) {
    tm["advance"]->start(); 

    size_t nb_stencils = grid_ref.getStencilsSize(); 
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    // backup the current solution so we can perform intermediate steps
    std::vector<double> original_solution = this->U_G; 
    std::vector<double>& s = this->U_G; 
    std::vector<SolutionType> feval1(nb_stencils);  

    this->solve(s, &feval1, cur_time); 

    tm["applyDerivsONLY"]->start();  
    // TODO: (ADD) Use 5 point Cartesian formula for the Laplacian
    //lapl_deriv = grid.computeCartLaplacian(s);
    
#if 1
    for (int i = 0; i < feval1.size(); i++) {
        Vec3& v = nodes[i];
        printf("(local: %d), lapl(%f,%f,%f)= %f\tUpdated Solution=%f\n", i, v.x(), v.y(),v.z(), feval1[i], s[i]);
    }
#endif

    // compute u^* = u^n + dt*lapl(u^n)
    // explicit scheme
    for (int i = 0; i < feval1.size(); i++) {
        // first order
        Vec3& v = nodes[i];
        //printf("dt= %f, time= %f\n", dt, time);
        double f = 0.;//force(i, v, time*dt);
        //double f = 0.;
        //printf("force (local: %d) = %f\n", i, f);
        // TODO: offload this to GPU. 
        // s += alpha * lapl_deriv + f
        // NOTE: f is updated each timestep
        //       lapl_deriv is calculated on GPU (ideally by passing a GPU mem pointer into it) 
        s[i] = s[i] + dt* ( feval1[i] + f);
    }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time); 

   tm["applyDerivsONLY"]->stop(); 
#if 0
    // Now we need to make sure all CPUs have R2 (intermediate part of timestep)
    // Do NOT use Domain as buffer for computation
    for (int i = 0; i < s1.size(); i++) {
        subdomain->U_G[i] = s1[i];
    }
#endif 
    comm_ref.broadcastObjectUpdates(this);
}


//----------------------------------------------------------------------

// Advance the equation one time step using the Domain class to perform communication
// Depends on Constructor #2 to be used so that a Domain class exists within this class.
void TimeDependentPDE::advanceSecondEuler(double dt)
{
#if 0
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

    tm["advance"]->start(); 

    // backup the current solution so we can perform intermediate steps
    vector<double> original_solution = this->U_G; 

    // This requires two steps: 
    vector<double> s;
    vector<double> s1;

    // Assume our solution vector contains all updates necessary.
    // comm_unit->broadcaseObjectUpdates(this); 

    // Only go up to the number of stencils since we solve for a subset of the values in U_G
    // Since U_G in R is at end of U_G vector we can ignore those.
    for (int i = 0; i < s.size(); i++) {
        s[i] = this->U_G[i];
    }

    // This is on the CPU or GPU depending on type of Derivative class used
    // (e.g., DerivativeCL will compute on GPU using OpenCL)
    der->applyWeightsForDeriv(RBFFD::LAPL, s, lapl_deriv);

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
    der->applyWeightsForDeriv(RBFFD::LAPL, s1, lapl_deriv);

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
#endif 
}

