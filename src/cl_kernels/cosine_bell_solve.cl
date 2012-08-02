#ifndef __COSINE_BELL_SOLVE_H__
#define __COSINE_BELL_SOLVE_H__

#include "useDouble.cl"
#include "applyWeights.cl"
#include "cart2sph.cl"

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
            uint stencil_padded_size, 
            int useHyperviscosity
            )
{
        FLOAT pi = acos(-1.f);
        FLOAT alpha =pi/2.f;
        FLOAT a = 1.f;//6.37122e6f; // radius of earth in meters
        FLOAT R = a/3.f;
        FLOAT u0 = 2.f*pi*a/1036800.f; // The initial velocity (scalar in denom is 12days in seconds)


        FLOAT dh_dlambda= applyWeights(lambda_weights, u_t, indx, stencils, stencil_size, stencil_padded_size);
        FLOAT dh_dtheta = applyWeights(theta_weights, u_t, indx, stencils, stencil_size, stencil_padded_size);

//        __global FLOAT4* node = nodes[indx];
        FLOAT4 spherical_coords = cart2sph(nodes[indx]);
        // longitude, latitude respectively:
        FLOAT lambda = spherical_coords.x;
        FLOAT theta = spherical_coords.y;

        FLOAT vel_u =   u0 * (cos(theta) * cos(alpha) + sin(theta) * cos(lambda) * sin(alpha));
        //double vel_v = - u0 * (cos(lambda) * sin(alpha));
        FLOAT vel_v = - u0 * (sin(lambda) * sin(alpha));

        // dh/dt + u / cos(theta) * dh/d(lambda) + v * dh/d(theta) = 0
        // dh/dt = - [diag(u/cos(theta)) * D_LAMBDA * h + diag(v/a) * D_THETA * h] + H
        //FLOAT f_out = -((vel_u/(a * cos(theta))) * dh_dlambda + (vel_v/a) * dh_dtheta);
        // NOTE: the 1/cos was analyticaly removed
        FLOAT f_out = -((vel_u/(a)) * dh_dlambda + (vel_v/a) * dh_dtheta);

        if (useHyperviscosity) {
                FLOAT hv_filter = applyWeights(hv_weights, u_t, indx, stencils, stencil_size, stencil_padded_size);
                // Filter is ONLY applied after the rest of the RHS is evaluated
                f_out += hv_filter;
        }

        return f_out;
}
// 2



FLOAT solve_block(__global FLOAT* u_t,
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
            uint stencil_padded_size,
            int useHyperviscosity,
            __local FLOAT* shared
            )
{
    uint lid = get_local_id(0); 
    uint block_size = get_local_size(0);

        FLOAT pi = acos(-1.f);
        FLOAT alpha =pi/2.f;
        FLOAT a = 1.f;//6.37122e6f; // radius of earth in meters
        FLOAT R = a/3.f;
        FLOAT u0 = 2.f*pi*a/1036800.f; // The initial velocity (scalar in denom is 12days in seconds)

        // Apply weights into shared memory. Note: the first element of each shared block
        // will contain the derivative value for the current stencil
        applyWeights_block(lambda_weights, u_t, indx, stencils, stencil_size, stencil_padded_size, &shared[0]);
        applyWeights_block(theta_weights, u_t, indx, stencils, stencil_size, stencil_padded_size, &shared[block_size]);
        if (useHyperviscosity) {
            applyWeights_block(hv_weights, u_t, indx, stencils, stencil_size, stencil_padded_size, &shared[2*block_size]);
        }

        barrier(CLK_LOCAL_MEM_FENCE); 

        if (lid == 0) {
            FLOAT dh_dlambda = shared[0]; 
            FLOAT dh_dtheta = shared[block_size]; 

            FLOAT4 spherical_coords = cart2sph(nodes[indx]);
            // longitude, latitude respectively:
            FLOAT lambda = spherical_coords.x;
            FLOAT theta = spherical_coords.y;

            FLOAT vel_u =   u0 * (cos(theta) * cos(alpha) + sin(theta) * cos(lambda) * sin(alpha));
            //double vel_v = - u0 * (cos(lambda) * sin(alpha));
            FLOAT vel_v = - u0 * (sin(lambda) * sin(alpha));

            // dh/dt + u / cos(theta) * dh/d(lambda) + v * dh/d(theta) = 0
            // dh/dt = - [diag(u/cos(theta)) * D_LAMBDA * h + diag(v/a) * D_THETA * h] + H
            //FLOAT f_out = -((vel_u/(a * cos(theta))) * dh_dlambda + (vel_v/a) * dh_dtheta);
            // NOTE: the 1/cos was analyticaly removed
            FLOAT f_out = -((vel_u/(a)) * dh_dlambda + (vel_v/a) * dh_dtheta);

            if (useHyperviscosity) {
                    FLOAT hv_filter = shared[2*block_size]; 
                    // Filter is ONLY applied after the rest of the RHS is evaluated
                    f_out += hv_filter;
            }

            // We overwrite the first element of shared memory now that we have
            // computed everything
            shared[0] = f_out;
        }
}



#endif 

// These are included backwards because we want to define a general "solve" and
// "solve_block" routine that can be used by any solver for any test problem. Sort
// of a symbolic approacH
#include "rk4_warp_per_stencil.cl"
#include "euler_general.cl"
