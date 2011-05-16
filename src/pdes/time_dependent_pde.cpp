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
            s[grid_ref.g2l(*Q_iter)] = exactSolution->at(v, cur_time); 
        }
    }
    //    printf("============ End Initial Conditions ===========\n");
}

// Advancing requires: 
//  - computing an update to the current solution (i.e., calling applyWeightsForDerivs(currentSolution)) 
//  - applying the updates to the current solution (i.e., RK45 weighted summation of intermediate updates).
void TimeDependentPDE::advance(TimeScheme which, double delta_t) {
    tm["advance"]->start(); 
    switch (which) 
    {
        case EULER: 
            advanceFirstOrderEuler(delta_t); 
            break; 
        case MIDPOINT: 
            advanceSecondOrderMidpoint(delta_t);
            break;  
        case RK4: 
            advanceRungeKutta4(delta_t); 
            break;
        default: 
            std::cout << "[TimeDependentPDE] Invalid TimeScheme specified. Bailing...\n";
            exit(EXIT_FAILURE); 
            break; 
    };
    cur_time += delta_t; 
    tm["advance"]->stop(); 
}

void TimeDependentPDE::advanceFirstOrderEuler(double dt) {

    size_t nb_stencils = grid_ref.getStencilsSize(); 
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    // backup the current solution so we can perform intermediate steps
    std::vector<double> original_solution = this->U_G; 
    std::vector<double>& s = this->U_G; 
    std::vector<SolutionType> feval1(nb_stencils);  

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 

    // This routine will apply our weights to "s" in however many intermediate steps are required
    // and store the results in feval1
    this->solve(s, &feval1, cur_time); 

    // TODO: (ADD) Use 5 point Cartesian formula for the Laplacian
    //lapl_deriv = grid.computeCartLaplacian(s);

    // compute u^* = u^n + dt*lapl(u^n)
    for (size_t i = 0; i < feval1.size(); i++) {
        NodeType& v = nodes[i];
        //printf("dt= %f, time= %f\n", dt, time);
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        s[i] = s[i] + dt* ( feval1[i] + f);
        //printf("(local: %lu), lapl(%f,%f,%f)= %f\tInput Solution=%f\n", i, v.x(), v.y(),v.z(), feval1[i], s[i]);
    }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time); 

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
void TimeDependentPDE::advanceSecondOrderMidpoint(double dt)
{
    size_t nb_stencils = grid_ref.getStencilsSize(); 
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    // backup the current solution so we can perform intermediate steps
    std::vector<double> original_solution = this->U_G; 
    std::vector<double>& s = this->U_G; 

    // Midpoint requires two evaluations: f(y(t)) and f(y + (h/2) * f(y(t)))
    std::vector<SolutionType> feval1(nb_stencils);  
    std::vector<SolutionType> feval2(nb_stencils);  

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 

    // This routine will apply our weights to "s" in however many intermediate steps are required
    // and store the results in feval1
    this->solve(s, &feval1, cur_time); 

    // FIXME: we dont have a way to send updates for general vectors (YET)
    // In the meantime, we overwrite solutions with updates and broadcast the
    // update for intermediate steps. 
    // compute u^* = u^n + dt*lapl(u^n)
    for (size_t i = 0; i < feval1.size(); i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * feval1 
        s[i] +=  0.5* dt * ( feval1[i] + f);
    }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+0.5*dt); 

    // leverage auto-transmit updates for U_G (because vector s is a reference to U_G
    comm_ref.broadcastObjectUpdates(this);

    // y*(t) = y(t) + h * feval2
    // but feval2 = lapl[ y(t) + h/2 * feval1 ] (content between [..] was computed above)
    this->solve(s, &feval2, cur_time+0.5*dt); 

    // Finish step for midpoint method
    for (size_t i = 0; i < feval2.size(); i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * feval1 
        s[i] = original_solution[i] + dt * ( feval2[i] + f);
    }

    // Make sure any boundary conditions are met. 
    this->enforceBoundaryConditions(s, cur_time+dt);
    
    // Ensure we have consistent values across the board
    comm_ref.broadcastObjectUpdates(this);
}

void TimeDependentPDE::advanceRungeKutta4(double dt) 
{
    size_t nb_stencils = grid_ref.getStencilsSize(); 
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    // backup the current solution so we can perform intermediate steps
    std::vector<double> original_solution = this->U_G; 
    std::vector<double>& s = this->U_G; 

    std::vector<SolutionType> k1(nb_stencils); // f(t_n, y_n)
    std::vector<SolutionType> k2(nb_stencils); // f(t_n + 0.5dt, y_n + 0.5dt*k1)
    std::vector<SolutionType> k3(nb_stencils); // f(t_n + 0.5dt, y_n + 0.5dt*k2) 
    std::vector<SolutionType> k4(nb_stencils); // f(t_n + dt, y_n + dt*k3) 

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 


    // ------------------- K1 ------------------------
    // This routine will apply our weights to "s" in however many intermediate steps are required
    // and store the results in k1 
    this->solve(s, &k1, cur_time); 

    // FIXME: we dont have a way to send updates for general vectors (YET)
    // In the meantime, we overwrite solutions with updates and broadcast the
    // update for intermediate steps. 
    // compute u^* = u^n + dt*lapl(u^n)
    s = k1; 
    // This will force all procs to get updates for k1 (distributed)
    comm_ref.broadcastObjectUpdates(this); 

    // Backup the complete k1
    k1 = s; 

    // ------------------- K2 ------------------------
    for (size_t i = 0; i < k1.size(); i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = 0.5*dt * ( k1[i] + f);
    }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+0.5*dt); 

    // leverage auto-transmit updates for U_G (because vector s is a reference to U_G
    comm_ref.broadcastObjectUpdates(this);

    // y*(t) = y(t) + h * k2
    // but k2 = lapl[ y(t) + h/2 * k1 ] (content between [..] was computed above)
    this->solve(s, &k2, cur_time+0.5*dt); 

    s = k2;
    comm_ref.broadcastObjectUpdates(this); 
    k2 = s;

    // ------------------- K3 ------------------------
    for (size_t i = 0; i < k2.size(); i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = 0.5*dt * ( k2[i] + f);
    }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+0.5*dt); 

    // leverage auto-transmit updates for U_G (because vector s is a reference to U_G
    comm_ref.broadcastObjectUpdates(this);

    this->solve(s, &k3, cur_time+0.5*dt); 

    s = k3;
    comm_ref.broadcastObjectUpdates(this); 
    k3 = s;

    // ------------------- K4 ------------------------
    for (size_t i = 0; i < k3.size(); i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = dt * ( k3[i] + f);
    }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+dt); 

    // leverage auto-transmit updates for U_G (because vector s is a reference to U_G
    comm_ref.broadcastObjectUpdates(this);

    this->solve(s, &k4, cur_time+dt); 

    s = k4;
    comm_ref.broadcastObjectUpdates(this); 
    k4 = s;

    // ------------------- K4 ------------------------
    // FINAL STEP: y_n+1 = y_n + 1/6 * h * (k1 + 2*k2 + 2*k3 + k4)
    //
    for (size_t i = 0; i < k4.size(); i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * feval1 
        s[i] = original_solution[i] + (dt/6.) * ( k1[i] + 2.*k2[i] + 2.*k3[i] + k4[i]);
    }

    // Make sure any boundary conditions are met. 
    this->enforceBoundaryConditions(s, cur_time+dt);
    
    // Ensure we have consistent values across the board
    comm_ref.broadcastObjectUpdates(this);
}
