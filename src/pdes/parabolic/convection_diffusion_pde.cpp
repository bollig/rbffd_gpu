#include "convection_diffusion_pde.h"

void ConvectionDiffusionPDE::setupTimers()
{
    ;
}
#if 0
void ConvectionDiffusionPDE::fillBoundaryConditions(ExactSolution* exact) {
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


// We want the initial conditions to be filled using the exact solution
// as well as the boundary conditions to be exact
void ConvectionDiffusionPDE::fillInitialConditions(ExactSolution* exact) {
    this->TimeDependentPDE::fillInitialConditions(exact);
    this->fillBoundaryConditions(exact);
    exact_ptr = exact;
}


// Get the time dependent diffusion coeffs 
void ConvectionDiffusionPDE::fillDiffusion(std::vector<SolutionType>& diff, double t) {
    for(size_t i = 0; i < diff.size(); i++) {
        NodeType& pt = grid_ref.getNode(i); 
        diff[i] = exact_ptr->diffuseCoefficient(pt, t);
    }
}

// Handle the boundary conditions however we want to. 
// NOTE: for this PDE we assume there is no influx of heat on the boundary
// FIXME: the PDE is not 0 on the boundary for a regular grid. 
void ConvectionDiffusionPDE::enforceBoundaryConditions(std::vector<SolutionType>& u_t, double t)
{
    size_t nb_bnd = grid_ref.getBoundaryIndicesSize(); 
    std::vector<size_t>& bnd_index = grid_ref.getBoundaryIndices();
    std::vector<NodeType>& nodes = grid_ref.getNodeList(); 

    for (int i = 0; i < nb_bnd; i++) {
        // first order
        NodeType& v = nodes[bnd_index[i]];
        //            printf("bnd[%d] = {%ld} %f, %f, %f\n", i, bnd_index[i], v.x(), v.y(), v.z());
        // u_t[bnd_index[i]] = boundary_values[i];
        u_t[bnd_index[i]] = exact_ptr->at(v, t); 
    }
}

#endif 


// Get the convection scalars/vectors for each point
void ConvectionDiffusionPDE::fillConvection(std::vector<SolutionType>& diff, double t) {
    for(size_t i = 0; i < diff.size(); i++) {
        NodeType& pt = grid_ref.getNode(i); 
        diff[i] = exact_ptr->convectionCoefficient(pt, t);
    }
}

void ConvectionDiffusionPDE::assemble() 
{
    if (!weightsPrecomputed) {
        der_ref.computeAllWeightsForAllStencils();
    }
}

// evaluate f_out = f(t,u(t)) so we can use it to compute a 
// timestep: u(t+h) = u(t) + h*f(t,u(t))
// For the diffusion equation this is f(t,u(t)) = laplacian(u(t))
// FIXME: we are not using a time-based diffusion coefficient. YET. 
void ConvectionDiffusionPDE::solve(std::vector<SolutionType>& u_t, std::vector<SolutionType>* f_out, size_t n, double t)
{
    std::vector<SolutionType> u_lapl_deriv(n); 
    std::vector<SolutionType> K_dot_lapl_U(n); 
    std::vector<SolutionType> diffusion(n);  

    if (splitLaplacian) {
        std::cout << "[ConvectionDiffusionPDE] todo: finish split laplacian. analytic derivs of RBFs need to be updated, plus new RHS types for RBFFD created.\n";
        exit(EXIT_FAILURE); 
    } else {
        der_ref.applyWeightsForDeriv(RBFFD::LAPL, u_t, u_lapl_deriv); 
    }

    // Get the diffusivity of the domain for at the current time
    this->fillDiffusion(diffusion, u_t, t);

    for (size_t i = 0; i < n; i++) {
        K_dot_lapl_U[i] = diffusion[i] * u_lapl_deriv[i]; 
    }

    if (uniformDiffusion) {
        for (size_t i = 0; i < n; i++) {
#if 0
            SolutionType exact = exact_ptr->diffuseCoefficient(grid_ref.getNode(i), t) * exact_ptr->laplacian(grid_ref.getNode(i),t);
            double error = fabs(K_dot_lapl_U[i] - exact)/fabs(exact); 
          
            std::cout << "computed K_dot_lapl_U[" << i << "] = " << K_dot_lapl_U[i] << ", EXACT= " << exact; 
            std::cout << ((error > 1e-1) ? "***************" : ""); 
            std::cout << std::endl;
#endif 
            (*f_out)[i] = K_dot_lapl_U[i];
        }
    } else {
        // If we have non-uniform diffusion, more derivatives are requried
        std::vector<SolutionType> u_x_deriv(n); 
        std::vector<SolutionType> u_y_deriv(n); 
        std::vector<SolutionType> u_z_deriv(n); 

        std::vector<SolutionType> diff_x_deriv(n); 
        std::vector<SolutionType> diff_y_deriv(n); 
        std::vector<SolutionType> diff_z_deriv(n); 

        der_ref.applyWeightsForDeriv(RBFFD::X, u_t, u_x_deriv); 
        der_ref.applyWeightsForDeriv(RBFFD::Y, u_t, u_y_deriv); 
        der_ref.applyWeightsForDeriv(RBFFD::Z, u_t, u_z_deriv); 

        der_ref.applyWeightsForDeriv(RBFFD::X, diffusion, diff_x_deriv); 
        der_ref.applyWeightsForDeriv(RBFFD::Y, diffusion, diff_y_deriv); 
        der_ref.applyWeightsForDeriv(RBFFD::Z, diffusion, diff_z_deriv); 

        for (size_t i = 0; i < n; i++) {
            SolutionType grad_K[3] = { diff_x_deriv[i], diff_y_deriv[i], diff_z_deriv[i] }; 
            SolutionType grad_U[3] = { u_x_deriv[i], u_y_deriv[i], u_z_deriv[i] }; 

            SolutionType grad_K_dot_grad_U = grad_K[0] * grad_U[0] + grad_K[1] * grad_U[1] + grad_K[2] * grad_U[2];

#if 0
            NodeType& pt = grid_ref.getNode(i); 
            SolutionType lapl = grad_K_dot_grad_U + K_dot_lapl_U[i]; 
            Vec3 exact_diff_grad = exact_ptr->diffuseGradient(pt, t); 
            Vec3 exact_u_grad = exact_ptr->gradient(pt,t);
            SolutionType exact_dot = exact_diff_grad[0] * exact_u_grad[0] + exact_diff_grad[1] * exact_u_grad[1] + exact_diff_grad[2] * exact_u_grad[2]; 
            SolutionType exact = exact_dot + exact_ptr->diffuseCoefficient(pt, t) * exact_ptr->laplacian(pt,t);
            double error = fabs(lapl - exact)/fabs(exact); 
         
            //std::cout << "computed grad_K_dot_grad_U[" << i << "] = " << grad_K_dot_grad_U << ", K_dot_lapl_U = " << K_dot_lapl_U[i] << ", EXACT LAPL= " << exact; 
            std::cout << "computed lapl[" << i << "] = " << grad_K_dot_grad_U + K_dot_lapl_U[i] << ", EXACT LAPL= " << exact; 
            std::cout << ((error > 1e-1) ? "***************" : ""); 
            std::cout << std::endl;
#endif 
            (*f_out)[i] = grad_K_dot_grad_U + K_dot_lapl_U[i];
        }
    }
}

