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
    std::cout << "[HeatPDE] done with initial conditions.\n";
}


// We want the initial conditions to be filled using the exact solution
// as well as the boundary conditions to be exact
void HeatPDE::fillInitialConditions(ExactSolution* exact) {
    this->TimeDependentPDE::fillInitialConditions(exact);
    this->fillBoundaryConditions(exact);
    exact_ptr = exact;
}


// Get the time dependent diffusion coeffs 
void HeatPDE::fillDiffusion(std::vector<SolutionType>& diff, std::vector<SolutionType>& sol, double t, size_t n_nodes) {
    for(size_t i = 0; i < n_nodes; i++) {
        NodeType& pt = grid_ref.getNode(i); 
        diff[i] = exact_ptr->diffuseCoefficient(pt, sol[i], t);
    }
}


void HeatPDE::assemble() 
{
    if (!weightsPrecomputed) {
        der_ref.computeAllWeightsForAllStencils();
    }
}

// evaluate f_out = f(t,u(t)) so we can use it to compute a 
// timestep: u(t+h) = u(t) + h*f(t,u(t))
// For the diffusion equation this is f(t,u(t)) = laplacian(u(t))
// FIXME: we are not using a time-based diffusion coefficient. YET. 
void HeatPDE::solve(std::vector<SolutionType>& u_t, std::vector<SolutionType>* f_out, size_t n_stencils, size_t n_nodes, double t)
{   
    // EFB06092011: div_grad is noticeably faster and it works. Gordon keeps
    // saying his didnt. I need to get a non-uniform test case with known exact
    // solution to verify that statement.
#define SOLVE_DIV_GRAD 0
#if SOLVE_DIV_GRAD
    this->solveDivGrad(u_t, f_out, n_stencils, n_nodes, t);
#else 
    this->solveRewrittenLaplacian(u_t, f_out, n_stencils, n_nodes, t);
#endif 
}

void HeatPDE::solveDivGrad(std::vector<SolutionType>& u_t, std::vector<SolutionType>* f_out, size_t n_stencils, size_t n_nodes, double t)
{
    // x,y,z components of the gradient of u_t
    std::vector<SolutionType> grad_u_x(n_nodes); 
    std::vector<SolutionType> grad_u_y(n_nodes);
    std::vector<SolutionType> grad_u_z(n_nodes);

    // diffusion coeffs 
    std::vector<SolutionType> diffusion(n_nodes);  
    
    // Equation is \div{K \cdot \grad{u}}
    std::vector<SolutionType> K_dot_grad_u_u(n_stencils);  

    // x,y,z components of divergence of (K \cdot \grad{u_t})
    std::vector<SolutionType> div_grad_u_x(n_stencils); 
    std::vector<SolutionType> div_grad_u_y(n_stencils);
    std::vector<SolutionType> div_grad_u_z(n_stencils);

    // ApplyWeights Parameters: type, input (size n_nodes), output (size n_stencils), update on gpu?

    der_ref.applyWeightsForDeriv(RBFFD::X, u_t, grad_u_x, true); 
    der_ref.applyWeightsForDeriv(RBFFD::Y, u_t, grad_u_y, true); 
    der_ref.applyWeightsForDeriv(RBFFD::Z, u_t, grad_u_z, true); 

    // Get the diffusivity of the domain for at the current time
    this->fillDiffusion(diffusion, u_t, t, n_nodes);

    // NEED TO GET GRAD and diffusion for ghost nodes here: 
    this->sendrecvUpdates(grad_u_x, "grad_u_x"); 
    this->sendrecvUpdates(grad_u_y, "grad_u_y"); 
    this->sendrecvUpdates(grad_u_z, "grad_u_z"); 
//Filled n_nodes above:    this->sendrecvUpdates(diffusion, "diffusion"); 

    // Compute K dot grad{u}
    // FIXME: we assume scalar diffusion, make this 
    for (size_t i = 0; i < n_nodes; i++) {
        grad_u_x[i] *= diffusion[i];
        grad_u_y[i] *= diffusion[i];
        grad_u_z[i] *= diffusion[i];
    }

    // Get divergence of quanity (K dot grad{u})
    der_ref.applyWeightsForDeriv(RBFFD::X, grad_u_x, div_grad_u_x, true); 
    der_ref.applyWeightsForDeriv(RBFFD::Y, grad_u_y, div_grad_u_y, true); 
    der_ref.applyWeightsForDeriv(RBFFD::Z, grad_u_z, div_grad_u_z, true); 

    // Finish computing divergence
    for (size_t i = 0; i < n_stencils; i++) {
        (*f_out)[i] = div_grad_u_x[i] + div_grad_u_y[i] + div_grad_u_z[i]; 
    }
}


void HeatPDE::solveRewrittenLaplacian(std::vector<SolutionType>& u_t, std::vector<SolutionType>* f_out, size_t n_stencils, size_t n_nodes,  double t)
{
    std::vector<SolutionType> u_lapl_deriv(n_stencils); 
    std::vector<SolutionType> K_dot_lapl_U(n_stencils); 
    std::vector<SolutionType> diffusion(n_nodes);  

#if 0
    // EFB06012011: need to add this test for splitting the lapl(u) into
    // separate d/dx(d/dx(u)), d/dy(d/dy) ops
    if (splitLaplacian) {
        std::cout << "[HeatPDE] todo: finish split laplacian. analytic derivs of RBFs need to be updated, plus new RHS types for RBFFD created.\n";
        exit(EXIT_FAILURE); 
    } else 
#endif 
    {
        der_ref.applyWeightsForDeriv(RBFFD::LAPL, u_t, u_lapl_deriv); 
    }

    // Get the diffusivity of the domain for at the current time NOTE: if we
    // assume that diffusion is a known across all subdomains then we can
    // bypass the partial fill and synchronization point (below, see
    // LABEL:DIFF_SYNC) and get the diffusion for all n_nodes here. What about
    // the case when diffusion is a function of the current solution? We assume
    // that u_t will have all n_nodes solutions synchronized before entering
    // this.
    this->fillDiffusion(diffusion, u_t, t, n_nodes); //n_stencils);

    for (size_t i = 0; i < n_stencils; i++) {
        K_dot_lapl_U[i] = diffusion[i] * u_lapl_deriv[i]; 
    }
    
#if 0
    // EFB06012011
    // To compare with the div_grad version we want to avoid shortcuts
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
    } else 
#endif 
    {
        // If we have non-uniform diffusion, more derivatives are requried
        std::vector<SolutionType> u_x_deriv(n_stencils); 
        std::vector<SolutionType> u_y_deriv(n_stencils); 
        std::vector<SolutionType> u_z_deriv(n_stencils); 

        std::vector<SolutionType> diff_x_deriv(n_stencils); 
        std::vector<SolutionType> diff_y_deriv(n_stencils); 
        std::vector<SolutionType> diff_z_deriv(n_stencils); 

        der_ref.applyWeightsForDeriv(RBFFD::X, u_t, u_x_deriv); 
        der_ref.applyWeightsForDeriv(RBFFD::Y, u_t, u_y_deriv); 
        der_ref.applyWeightsForDeriv(RBFFD::Z, u_t, u_z_deriv); 

        // LABEL:DIFF_SYNC
        //this->sendrecvUpdates(diffusion, "diffusion"); 

        der_ref.applyWeightsForDeriv(RBFFD::X, diffusion, diff_x_deriv); 
        der_ref.applyWeightsForDeriv(RBFFD::Y, diffusion, diff_y_deriv); 
        der_ref.applyWeightsForDeriv(RBFFD::Z, diffusion, diff_z_deriv); 

        for (size_t i = 0; i < n_stencils; i++) {
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


// Handle the boundary conditions however we want to. 
// NOTE: for this PDE we assume there is no influx of heat on the boundary
// FIXME: the PDE is not 0 on the boundary for a regular grid. 
void HeatPDE::enforceBoundaryConditions(std::vector<SolutionType>& u_t, double t)
{
    size_t nb_bnd = grid_ref.getBoundaryIndicesSize(); 
    std::vector<size_t>& bnd_index = grid_ref.getBoundaryIndices();
    std::vector<NodeType>& nodes = grid_ref.getNodeList(); 

    for (int i = 0; i < nb_bnd; i++) {
        // first order
        NodeType& v = nodes[bnd_index[i]];
       // printf("bnd[%d] = {%ld} %f, %f, %f\n", i, bnd_index[i], v.x(), v.y(), v.z());
        // u_t[bnd_index[i]] = boundary_values[i];
        u_t[bnd_index[i]] = exact_ptr->at(v, t); 
    }
}

