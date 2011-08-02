
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
            int useHyperviscosity
            )
{
        FLOAT pi = 3.14159;
        FLOAT R = 1./3.;
        FLOAT alpha =-pi/2.;
        FLOAT a = 6.37122e6; // radius of earth in meters
        FLOAT u0 = 2*pi*a/1036800.; // The initial velocity (scalar in denom is 12days in seconds)

        FLOAT dh_dlambda= applyWeights(lambda_weights, u_t, indx, stencils, stencil_size);
        FLOAT dh_dtheta = applyWeights(theta_weights, u_t, indx, stencils, stencil_size);

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
        FLOAT f_out = -((vel_u/(a * cos(theta))) * dh_dlambda + (vel_v/a) * dh_dtheta);

        if (useHyperviscosity) {
                FLOAT hv_filter = applyWeights(hv_weights, u_t, indx, stencils, stencil_size);
                // Filter is ONLY applied after the rest of the RHS is evaluated
                f_out += hv_filter;
        }

        return f_out;
}
