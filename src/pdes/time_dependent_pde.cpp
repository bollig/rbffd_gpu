#include "time_dependent_pde.h"

#include <iostream>

void TimeDependentPDE::fillInitialConditions(ExactSolution* exactSolution) {

    exact_ptr = exactSolution;
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
//  - applying the updates to the current solution (i.e., RK4 weighted summation of intermediate updates).
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

    unsigned int nb_stencils = grid_ref.getStencilsSize(); 
    unsigned int nb_nodes = grid_ref.getNodeListSize(); 
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
    this->solve(s, &feval1, nb_stencils, nb_nodes, cur_time); 

    // compute u^* = u^n + dt*lapl(u^n)
    for (unsigned int i = 0; i < nb_nodes; i++) {
        NodeType& v = nodes[i];
        //printf("dt= %f, time= %f\n", dt, time);
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        s[i] = s[i] + dt* ( feval1[i] + f);
   }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time); 

    //comm_ref.broadcastObjectUpdates(this);
    this->sendrecvUpdates(s, "U_G");
}


//----------------------------------------------------------------------

// Advance the equation one time step using the Domain class to perform communication
// Depends on Constructor #2 to be used so that a Domain class exists within this class.
void TimeDependentPDE::advanceSecondOrderMidpoint(double dt)
{
    unsigned int nb_stencils = grid_ref.getStencilsSize(); 
    unsigned int nb_nodes = grid_ref.getNodeListSize(); 
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    // backup the current solution so we can perform intermediate steps
    std::vector<double> s(nb_nodes, 0.);

    // Midpoint requires two evaluations: f(y(t)) and f(y + (h/2) * f(y(t)))
    std::vector<SolutionType> feval1(nb_stencils);  
    std::vector<SolutionType> feval2(nb_stencils);  

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 

    // This routine will apply our weights to "s" in however many intermediate steps are required
    // and store the results in feval1
    this->solve(this->U_G, &feval1, nb_stencils, nb_nodes, cur_time); 

    // compute u^* = u^n + dt*lapl(u^n)
    for (unsigned int i = 0; i < nb_stencils; i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * feval1  @ t + h/2
        s[i] = this->U_G[i] + 0.5* dt * ( feval1[i] + f);
    }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+0.5*dt); 

    // Make sure our neighboring CPUs are aware of changes to values in set O (output)
    this->sendrecvUpdates(s, "s_intermediate");

    // y*(t) = y(t) + h * feval2
    // but feval2 = lapl[ y(t) + h/2 * feval1 ] @ t+0.5dt (content between [..]'s was computed above)
    this->solve(s, &feval2, nb_stencils, nb_nodes, cur_time+0.5*dt); 

    // Finish step for midpoint method
    for (unsigned int i = 0; i < nb_stencils; i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * feval1 
        U_G[i] = U_G[i] + dt * ( feval2[i] + f);
    }

    // Make sure any boundary conditions are met. 
    this->enforceBoundaryConditions(U_G, cur_time+dt);
    
    // Ensure we have consistent values across the board
    this->sendrecvUpdates(U_G, "U_G");
}

void TimeDependentPDE::advanceRungeKutta4(double dt) 
{
    unsigned int nb_stencils = grid_ref.getStencilsSize(); 
    unsigned int nb_nodes = grid_ref.getNodeListSize(); 
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    // backup the current solution so we can perform intermediate steps
    std::vector<double>& input_sol = this->U_G; 

    std::vector<double> s(nb_nodes, 0.);

    std::vector<SolutionType> k1(nb_stencils,0.); // f(t_n, y_n)
    std::vector<SolutionType> k2(nb_stencils,0.); // f(t_n + 0.5dt, y_n + 0.5dt*k1)
    std::vector<SolutionType> k3(nb_stencils,0.); // f(t_n + 0.5dt, y_n + 0.5dt*k2) 
    std::vector<SolutionType> k4(nb_stencils,0.); // f(t_n + dt, y_n + dt*k3) 

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 

//NOTE: between *****'s are kernel and comm. 
//*********
    // ------------------- K1 ------------------------
    // This routine will apply our weights to "s" in however many intermediate steps are required
    // and store the results in k1 
    this->solve(input_sol, &k1, nb_stencils, nb_nodes, cur_time); 
    //this->sendrecvUpdates(k1, "k1");

    // ------------------- K2 ------------------------
    for (unsigned int i = 0; i < nb_stencils; i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = input_sol[i] + 0.5*dt * ( k1[i] + f);
    }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+0.5*dt); 

    // FIXME: might not be necessary unless we strictly believe that enforcing
    // BC can only be done by controlling CPU
    // NOTE: increases s from nb_stencils to nb_nodes
    this->sendrecvUpdates(s, "s");
//*********
    // y*(t) = y(t) + h * k2
    // but k2 = lapl[ y(t) + h/2 * k1 ] (content between [..] was computed above)
    this->solve(s, &k2, nb_stencils, nb_nodes, cur_time+0.5*dt); 
    //    this->sendrecvUpdates(k2, "k2");

    // ------------------- K3 ------------------------
    for (unsigned int i = 0; i < nb_stencils; i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = input_sol[i] + 0.5*dt * ( k2[i] + f);
    }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+0.5*dt); 

    // FIXME: might not be necessary unless we strictly believe that enforcing
    // BC can only be done by controlling CPU
    this->sendrecvUpdates(s, "s");

//*********
    this->solve(s, &k3, nb_stencils, nb_nodes, cur_time+0.5*dt); 
    //    this->sendrecvUpdates(k3, "k3");

    // ------------------- K4 ------------------------
    for (unsigned int i = 0; i < nb_stencils; i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = input_sol[i] + dt * ( k3[i] + f);
    }

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+dt); 

    // FIXME: might not be necessary unless we strictly believe that enforcing
    // BC can only be done by controlling CPU
    this->sendrecvUpdates(s, "s");

//*********
    this->solve(s, &k4, nb_stencils, nb_nodes, cur_time+dt); 
   //    this->sendrecvUpdates(k4, "k4");

    // ------------------- FINAL ------------------------
    // FINAL STEP: y_n+1 = y_n + 1/6 * h * (k1 + 2*k2 + 2*k3 + k4)
    //
    for (unsigned int i = 0; i < nb_stencils; i++) {
        NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * feval1 
        input_sol[i] = input_sol[i] + (dt/6.) * ( k1[i] + 2.*k2[i] + 2.*k3[i] + k4[i]);
    }

    // Make sure any boundary conditions are met. 
    this->enforceBoundaryConditions(input_sol, cur_time+dt);
    
    // Ensure we have consistent values across the board
    this->sendrecvUpdates(input_sol, "U_G");
}
