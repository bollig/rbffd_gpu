#include "heat_pde_cl.h"

#include "rbffd/rbffd_cl.h"

void HeatPDE_CL::setupTimers()
{
;
}

void HeatPDE_CL::fillBoundaryConditions(ExactSolution* exact) {
    std::cout << "FILLING GPU BOUNDARY CONDITIONS\n";
    size_t nb_bnd = grid_ref.getBoundaryIndicesSize();
    std::vector<size_t>& bnd_index = grid_ref.getBoundaryIndices();
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    boundary_values.resize(nb_bnd);

    printf("Copying solution to bnd_sol (boundary solution buffer)\n"); 
    for (int i = 0; i < nb_bnd; i++) {
        NodeType& v = nodes[bnd_index[i]];
        boundary_values[i] = exact->at(v, 0.);
        // printf("boundary: %f %f %f ==> %f\n", v.x(), v.y(), v.z(),bnd_sol[i]);
    }
}


void HeatPDE_CL::fillInitialConditions(ExactSolution* exact) {
    this->TimeDependentPDE::fillInitialConditions(exact);
    this->fillBoundaryConditions(exact);
    exact_ptr = exact;
}


void HeatPDE_CL::assemble() 
{
    // Fill weights mat on GPU
}

void HeatPDE_CL::solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, size_t n, double t)
{
    // Call kernel to apply weights mat for single solution
}

void HeatPDE_CL::enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t)
{
    // Copy updated     
}

