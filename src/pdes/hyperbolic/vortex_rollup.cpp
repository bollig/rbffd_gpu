
#include "vortex_rollup.h"
#include "utils/geom/cart2sph.h"

// This should assemble a matrix L of weights which can be used to solve the PDE
void VortexRollup::assemble() {
    if (!weightsPrecomputed) {
        der_ref.computeAllWeightsForAllStencils();
    }

    static int assembled = 0; 
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
    double rho0 = 3.;
    double gamma = 5.;

    std::vector<SolutionType> dh_d_lambda(n_stencils);  
    
    der_ref.applyWeightsForDeriv(RBFFD::LAMBDA, u_t, dh_d_lambda, true); 

    for (unsigned int i = 0; i < n_stencils; i++) {
        NodeType& v = grid_ref.getNode(i);
        
        sph_coords_type spherical_coords = cart2sph(v.x(), v.y(), v.z());
        double theta_p = spherical_coords.phi; 
        double phi_p = spherical_coords.theta; 
        double temp = spherical_coords.r; 

        // From Natasha's email: 
        // 7/7/11 4:46 pm
        double rho_p = rho0 * cos(theta_p); 

        // NOTE: The sqrt(2) is written as sqrt(3.) in the paper with Grady.
        // Natasha verified sqrt(3) is required
        // Also, for whatever reason sech is not defined in the C standard
        // math. I provide it in the cart2sph header. 
        double Vt = (3.* sqrt(3.) / 2.) * (sech(rho_p) * sech(rho_p)) * tanh(rho_p); 
        double w_theta_P = (fabs(rho_p) < 4. * DBL_EPSILON) ? 0. : Vt / rho_p;

        // dh/dt + u' / cos(theta') * dh/d(lambda) = 0
        // u' = w(theta') cos(theta')
        // So, dh/dt = - w(theta_P) * dh/d(lambda)
        (*f_out)[i] = - (w_theta_P) * dh_d_lambda[i];
    }
    if (useHyperviscosity) {
        // Filter is ONLY applied after the rest of the RHS is evaluated
        std::vector<SolutionType> hv_filter; 
        der_ref.applyWeightsForDeriv(RBFFD::HV, u_t, hv_filter, true); 
        for (unsigned int i =0; i < n_stencils; i++) {
            (*f_out)[i] += hv_filter[i]; 
        }
    }
}

void VortexRollup::setupTimers() {

}



