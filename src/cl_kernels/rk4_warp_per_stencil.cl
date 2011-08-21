#include "useDouble.cl"
// Assume the "solver will include this instead of the other way around
//#include "solver.cl"
//1

// -----------------------------------------------------------
// 		RK4:
//    k1 = dt*func(DM_Lambda, DM_Theta, H, u, t, nodes, useHV);
//    k2 = dt*func(DM_Lambda, DM_Theta, H, u+0.5*F1, t+0.5*dt, nodes, useHV);
//    k3 = dt*func(DM_Lambda, DM_Theta, H, u+0.5*F2, t+0.5*dt, nodes, useHV);
//    k4 = dt*func(DM_Lambda, DM_Theta, H, u+F3, t+dt, nodes, useHV);
// -----------------------------------------------------------

// We do a little extra work here to avoid computing a SAXPY between evaluations
// The SAXPY is to compute the input to the next evaluation step (i.e., K2 <- solve(u+0.5*K1) )
__kernel void 
evaluateRK4_block(
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
    int useHyperviscosity,

    __local FLOAT* shared
)
{
    // This is our stencil index
    uint i = get_group_id(0);    
    // This is the real starting index for our stencil
    uint j = i + offset_to_set; 

    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

    // We operate in 1 full warp per stencil (trim off any extra threads)
    if (j < nb_stencils_to_compute) {   

        // Repeat until stencil is complete: 
        //      - apply weights for block_size elements of stencil
        //      - add output to shared memory register for thread
        // Sum block_size elements of shared memory (prefix_sum)

        // All threads in block will apply weights for a single stencil
        // resulting derivative is stored in shared memory 
        solve_block(     u_plus_scaled_k_in,
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
                         useHyperviscosity, 
                         shared
                     );

        // Master thread from warp will write global memory
        if (lid == 0) {

            FLOAT feval1 = shared[lid]; 

            k_out[j] = dt * feval1;
            u_plus_scaled_k_out[j] = u_in[j] + k_scale * dt * feval1;
        }
    }
}

// We'll re-use the existing kernels
#include "rk4_classic.cl"




