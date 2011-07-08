
#include "vortex_rollup.h"

// This should assemble a matrix L of weights which can be used to solve the PDE
void VortexRollup::assemble() {
    if (!weightsPrecomputed) {
        der_ref.computeAllWeightsForAllStencils();
    }
}


// This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
void VortexRollup::solve(std::vector<SolutionType>& u_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t) {

    // Fill DM with interpolation weights for each stencil
    // The stencils will interpolate the function value at the stencil center
    // using a weighted combination of values in the stencil. Seems
    // counter-intuitive, but once we can interpolate our values at the stencil
    // centers we can start approximating derivatives for the stencil center
    // with weighted combinations. 

    
    // dh/dt = - W * D * h
    // where W = diag(w(theta_p)) and 
    //       D = B A^-1
    // with B_i,j = cos(theta_p_i) cos(theta_p_j) * sin(lambda_i - lambda_j) (1/r)(d theta / dr)
    //


    std::vector<SolutionType> interpolated_solution(n_nodes,1.);  
    der_ref.applyWeightsForDeriv(RBFFD::INTERP, u_t, interpolated_solution, true); 

    for (unsigned int i = 0; i < n_stencils; i++) {
        (*f_out)[i] = interpolated_solution[i]; 
    }
}


void VortexRollup::advance(TimeScheme which, double delta_t) {

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
        s[i] = feval1[i]; //s[i] + delta_t* ( feval1[i] + f);
        printf("%f %f\n", feval1[i], exact_ptr->at(grid_ref.getNode(i), 0));
   }

    cur_time += delta_t; 
}


void VortexRollup::setupTimers() {

}



