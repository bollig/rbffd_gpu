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
    if (!weightsPrecomputed) {
        ((RBFFD_CL&)der_ref).computeAllWeightsForAllStencils();
    }
}

// evaluate f_out = f(t,y(t)) so we can use it to compute a 
// timestep: y(t+h) = y(t) + h*f(t,y(t))
// For the diffusion equation this is f(t,y(t)) = laplacian(y(t))
// FIXME: we are not using a time-based diffusion coefficient. YET. 
void HeatPDE_CL::solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, double t)
{
    // 1) Update solution on gpu with ghost nodes
    // 2) apply weights to solution /
    //der_ref.applyWeightsForDeriv(RBFFD::LAPL, y_t_gpu, lapl_deriv_gpu, true);
    std::vector<SolutionType>& lapl_deriv = *f_out;
    lapl_deriv.resize(y_t.size()); 

    // This is on the CPU or GPU depending on type of Derivative class used
    // (e.g., DerivativeCL will compute on GPU using OpenCL)
    der_ref.applyWeightsForDeriv(RBFFD::LAPL, y_t, lapl_deriv);


# if 0
    for (int i = 0; i < lapl_deriv.size(); i++) {
        lapl_deriv[i] = -lapl_deriv[i]; 
    }
#endif
}


// Handle the boundary conditions however we want to. 
// NOTE: for this PDE we assume there is no influx of heat on the boundary
// FIXME: the PDE is not 0 on the boundary for a regular grid. 
void HeatPDE_CL::enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t)
{
        size_t nb_bnd = grid_ref.getBoundaryIndicesSize(); 
        std::vector<size_t>& bnd_index = grid_ref.getBoundaryIndices();
        std::vector<NodeType>& nodes = grid_ref.getNodeList(); 

        for (int i = 0; i < nb_bnd; i++) {
            // first order
            NodeType& v = nodes[bnd_index[i]];
            //            printf("bnd[%d] = {%ld} %f, %f, %f\n", i, bnd_index[i], v.x(), v.y(), v.z());
           // y_t[bnd_index[i]] = boundary_values[i];
           y_t[bnd_index[i]] = exact_ptr->at(v, t); 
        }
}

