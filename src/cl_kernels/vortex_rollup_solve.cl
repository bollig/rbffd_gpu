// f
#ifndef __VORTEX_ROLLUP_SOLVE_H__
#define __VORTEX_ROLLUP_SOLVE_H__
#include "applyWeights.cl"
#include "cart2sph.cl"

#include "useDouble.cl"

FLOAT solve(__global FLOAT* u_t,

            unsigned int indx,
            double t,
            __global FLOAT4* nodes,
            __global uint* stencils,
            __global FLOAT* x_weights,
            __global FLOAT* y_weights,
            __global FLOAT* z_weights,
            __global FLOAT* lapl_weights,
            __global FLOAT* r_weights,
            __global FLOAT* lambda_weights,
            __global FLOAT* theta_weights,
            __global FLOAT* hv_weights,

            uint nb_nodes,
            uint stencil_size,
            int useHyperviscosity
            )
{
    FLOAT rho0 = 3.;
    FLOAT gamma = 5.;
    FLOAT mach_eps = 0.00000001;

        FLOAT dh_dlambda= applyWeights(lambda_weights, u_t, indx, stencils, stencil_size);

//        __global FLOAT4* node = nodes[indx];
        FLOAT4 spherical_coords = cart2sph(nodes[indx]);
        // longitude, latitude respectively:
//        FLOAT lambda = spherical_coords.x;
        FLOAT theta_p = spherical_coords.y;


        // From Natasha's email: 
        // 7/7/11 4:46 pm
        FLOAT rho_p = rho0 * cos(theta_p); 

        // NOTE: The sqrt(2) is written as sqrt(3.) in the paper with Grady.
        // Natasha verified sqrt(3) is required
        // Also, for whatever reason sech is not defined in the C standard
        // math. I provide it in the cart2sph header. 
        FLOAT Vt = (3.* sqrt(3.) / 2.) * (sech(rho_p) * sech(rho_p)) * tanh(rho_p); 
        
        FLOAT w_theta_P = (fabs(rho_p) < 4*mach_eps) ? 0. : Vt / rho_p;

        FLOAT f_out = - w_theta_P * dh_dlambda; 

        if (useHyperviscosity) {
                FLOAT hv_filter = applyWeights(hv_weights, u_t, indx, stencils, stencil_size);
                // Filter is ONLY applied after the rest of the RHS is evaluated
                f_out += hv_filter;
        }

        return f_out;
}

#endif 
