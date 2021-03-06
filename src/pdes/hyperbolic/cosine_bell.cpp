
#include "cosine_bell.h"
#include "utils/geom/cart2sph.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <map>
#include <fstream>
#include <typeinfo>


// This should assemble a matrix L of weights which can be used to solve the PDE
void CosineBell::assemble() {
    if (!weightsPrecomputed) {
        der_ref.computeAllWeightsForAllStencils();
        weightsPrecomputed=true;
    }
}


// This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
void CosineBell::solve(std::vector<SolutionType>& u_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t) {

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

        double vel_u =   u0 * (cos(theta) * cos(alpha) + sin(theta) * cos(lambda) * sin(alpha));
        //double vel_v = - u0 * (cos(lambda) * sin(alpha));
        double vel_v = - u0 * (sin(lambda) * sin(alpha));

        // dh/dt + u / cos(theta) * dh/d(lambda) + v * dh/d(theta) = 0
        // dh/dt = - [diag(u/cos(theta)) * D_LAMBDA * h + diag(v/a) * D_THETA * h] + H
        //(*f_out)[i] = -((vel_u/(a * cos(theta))) * dh_dlambda[i] + (vel_v/a) * dh_dtheta[i]);
        //NOTE: the 1/cos is analyticaly removed:
        (*f_out)[i] = -(     (vel_u/(a)) * dh_dlambda[i] + (vel_v/a) * dh_dtheta[i]     );

    }
    if (useHyperviscosity) {
        // Filter is ONLY applied after the rest of the RHS is evaluated
        for (unsigned int i =0; i < n_stencils; i++) {
            (*f_out)[i] += hv_filter[i];
        }
    }

#if 0
    std::ofstream fout;
    fout.open("output/f_out.mtx");
    for (size_t i = 0; i < n_stencils; i++) {
        fout << std::setprecision(10) << (*f_out)[i] << std::endl;
    }
    fout.close();
exit(-1);
#endif


}

void CosineBell::setupTimers() {

}



