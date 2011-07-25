
#include "cosine_bell.h"
#include "utils/geom/cart2sph.h"

// This should assemble a matrix L of weights which can be used to solve the PDE
void CosineBell::assemble() {
    if (!weightsPrecomputed) {
        der_ref.computeAllWeightsForAllStencils();
        weightsPrecomputed=true; 
    }
}


// This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
void CosineBell::solve(std::vector<SolutionType>& u_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t) {
    // RADIUS OF THE SPHERE: 
    double a = 1.;//6.37122*10^6; % radius of earth in meters
    // RADIUS OF THE BELL
    double R = a/3.;
    // ANGLE OF ADVECTION AROUND SPHERE (from Equator and Greenwhich Mean Time) 
    double alpha = -M_PI/2.;
    // The initial velocity (NOTE: scalar in denom is 12 days in seconds)
    double u0 = (2.*M_PI*a)/1036800.; 
    
    std::vector<SolutionType> dh_dlambda(n_stencils);  
    std::vector<SolutionType> dh_dtheta(n_stencils);  
    std::vector<SolutionType> hv_filter(n_stencils);  
    
    der_ref.applyWeightsForDeriv(RBFFD::LAMBDA, u_t, dh_dlambda, true); 
    der_ref.applyWeightsForDeriv(RBFFD::THETA, u_t, dh_dtheta, true); 
    der_ref.applyWeightsForDeriv(RBFFD::HV, u_t, hv_filter, true); 

    for (unsigned int i = 0; i < n_stencils; i++) {
        NodeType& v = grid_ref.getNode(i);
        
        sph_coords_type spherical_coords = cart2sph(v.x(), v.y(), v.z());
        // longitude, latitude respectively:
        double lambda = spherical_coords.theta; 
        double theta = spherical_coords.phi; 

        double vel_u =   u0 * (cos(theta) * cos(alpha) - sin(theta) * sin(lambda) * sin(alpha)); 
        double vel_v = - u0 * (cos(lambda) * sin(alpha));

        // dh/dt + u / cos(theta) * dh/d(lambda) + v * dh/d(theta) = 0
        // dh/dt = - [diag(u/cos(theta)) * D_LAMBDA * h + diag(v/a) * D_THETA * h] + H
        (*f_out)[i] = -((vel_u/(a * cos(theta))) * dh_dlambda[i] + (vel_v/a) * dh_dtheta[i]);
    
    }
    if (useHyperviscosity) {
        // Filter is ONLY applied after the rest of the RHS is evaluated
        for (unsigned int i =0; i < n_stencils; i++) {
            (*f_out)[i] += hv_filter[i]; 
        }
    }
}

void CosineBell::setupTimers() {

}



