#include "useDouble.cl"
#include "solver.cl"

// -----------------------------------------------------------
// 		CLASSIC RK4:
//
// -----------------------------------------------------------

// We do a little extra work here to avoid computing a SAXPY between evaluations
// The SAXPY is to compute the input to the next evaluation step (i.e., K2 <- solve(u+0.5*K1) )
__kernel void 
evaluateRK4_classic(
    __global FLOAT* u_in,
    __global FLOAT* u_plus_scaled_k_in,
    __global FLOAT* k_out,
    __global FLOAT* u_plus_scaled_k_out,
    FLOAT dt,
    FLOAT cur_time,
    FLOAT k_scale,

    // if we want to run this kernel on set QmD, offset is 0. To run kernel on set D, offset should be the offset in num elements to get to D
    uint offset_to_set,
    // This should not exceed the number of stencils in the set QmD, or set D
    uint nb_stencils_to_compute,

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
    uint i = get_global_id(0);
    uint j = i + offset_to_set;
    if(j < nb_stencils_to_compute) {
            FLOAT feval1 = solve(u_plus_scaled_k_in,
                                 j,
                                 cur_time,
                                 nodes,
                                 stencils,
                                 x_weights,
                                 y_weights,
                                 z_weights,
                                 lapl_weights,
                                 r_weights,
                                 lambda_weights,
                                 theta_weights,
                                 hv_weights,
                                 nb_nodes,
                                 stencil_size,
                                 useHyperviscosity
                                 );

            //    k1 = dt*func(DM_Lambda, DM_Theta, H, u, t, nodes, useHV);
            //    k2 = dt*func(DM_Lambda, DM_Theta, H, u+0.5*F1, t+0.5*dt, nodes, useHV);
            //    k3 = dt*func(DM_Lambda, DM_Theta, H, u+0.5*F2, t+0.5*dt, nodes, useHV);
            //    k4 = dt*func(DM_Lambda, DM_Theta, H, u+F3, t+dt, nodes, useHV);

            k_out[j] = dt * feval1;
            u_plus_scaled_k_out[j] = u_in[j] + k_scale * dt * feval1;

    }
}




// -----------------------------------------------------------
// Evaluate the final substep of RK4 and advance the solution
// -----------------------------------------------------------
__kernel void
advanceRK4_classic(
         __global FLOAT* u_in,
         __global FLOAT* unscaled_k1,
         __global FLOAT* unscaled_k2,
         __global FLOAT* unscaled_k3,
         __global FLOAT* unscaled_k4,
         __global FLOAT* u_out,

    // if we want to run this kernel on set QmD, offset is 0. To run kernel on set D, offset should be the offset in num elements to get to D
    uint offset_to_set,
    // This should not exceed the number of stencils in the set QmD, or set D
    uint nb_stencils_to_compute,

    // The rest of these should be unnecessary:
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
    uint i = get_global_id(0);
    uint j = i + offset_to_set;
    if(j < nb_stencils_to_compute) {
        FLOAT sol = u_in[j];

        // Note: k1 and k2 are scaled by 0.5, but we do NOT remove that scale.
        //       Instead we adjust the scalar in the equation below
        FLOAT k1 = unscaled_k1[j];
        FLOAT k2 = unscaled_k2[j];
        FLOAT k3 = unscaled_k3[j];
        FLOAT k4 = unscaled_k4[j];

        // See logic above.
        u_out[j] = sol + (k1 + 2.f*k2 + 2.f*k3 + k4) / 6.f;
    }
}



