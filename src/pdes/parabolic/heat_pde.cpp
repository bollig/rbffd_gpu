#include "heat_pde.h"

void HeatPDE::setupTimers()
{
;
}

void HeatPDE::fillBoundaryConditions(ExactSolution* exact) {
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


void HeatPDE::fillInitialConditions(ExactSolution* exact) {
    this->TimeDependentPDE::fillInitialConditions(exact);
    this->fillBoundaryConditions(exact);
}


void HeatPDE::assemble() 
{
    std::cout << "NEED TO RECOMPUTE WEIGHTS? ";
    der_ref.computeAllWeightsForAllStencils();
}

// evaluate f_out = f(t,y(t)) so we can use it to compute a 
// timestep: y(t+h) = y(t) + h*f(t,y(t))
// For the diffusion equation this is f(t,y(t)) = laplacian(y(t))
// FIXME: we are not using a time-based diffusion coefficient. YET. 
void HeatPDE::solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, double t)
{
    std::vector<SolutionType>& lapl_deriv = *f_out;
    lapl_deriv.resize(y_t.size()); 

    // This is on the CPU or GPU depending on type of Derivative class used
    // (e.g., DerivativeCL will compute on GPU using OpenCL)
    der_ref.applyWeightsForDeriv(RBFFD::LAPL, y_t, lapl_deriv);
}

// TODO: extend this class and compute diffusion in two terms: lapl(y(t)) = div(y(t)) .dot. grad(y(t))


// Handle the boundary conditions however we want to. 
// NOTE: for this PDE we assume there is no influx of heat on the boundary
// FIXME: the PDE is not 0 on the boundary for a regular grid. 
void HeatPDE::enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t)
{
        size_t nb_bnd = grid_ref.getBoundaryIndicesSize(); 
        std::vector<size_t>& bnd_index = grid_ref.getBoundaryIndices();
        std::vector<NodeType>& nodes = grid_ref.getNodeList(); 

        for (int i = 0; i < nb_bnd; i++) {
            // first order
            NodeType& v = nodes[bnd_index[i]];
            //            printf("bnd[%d] = {%ld} %f, %f, %f\n", i, bnd_index[i], v.x(), v.y(), v.z());
            y_t[bnd_index[i]] = boundary_values[i];
        }
}

