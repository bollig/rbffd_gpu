
#include "vortex_rollup.h"

// This should assemble a matrix L of weights which can be used to solve the PDE
void VortexRollup::assemble() {

}


// This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
void VortexRollup::solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t) {
    this->cur_time = t; 
    this->fillInitialConditions(exact_ptr);
//    exit(EXIT_FAILURE);
}

void VortexRollup::setupTimers() {

}



