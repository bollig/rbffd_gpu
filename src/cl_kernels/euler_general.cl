#include "useDouble.cl"
#include "constants.cl"
#include "solver.cl"

__kernel void
advanceEuler(

        __global FLOAT* solution_in,
        __global FLOAT* solution_out,
        // if we want to run this kernel on set QmD, offset is 0. To run kernel on set D, offset should be the offset in num elements to get to D
        uint offset_to_set,
        // This should not exceed the number of stencils in the set QmD, or set D
        uint nb_stencils_to_compute,
        float dt,
        float cur_time,

        __global FLOAT4* nodes,

        __global uint* stencils,

        __global FLOAT* x_weights,
        __global FLOAT* y_weights,
        __global FLOAT* z_weights,
        __global FLOAT* lapl_weights,
        __global FLOAT* r_weights,
        __global FLOAT* lambda_weights,
        __global FLOAT* theta_weights,
        __global FLOAT* hv_weights
        )
{
        uint i = get_global_id(0);
        uint j = i + offset_to_set;
        if(j < nb_stencils_to_compute) {
                // This routine will apply our weights to "s" in however many intermediate steps are required
                // and store the results in feval1
                FLOAT feval1 = solve(solution_in,
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
                                     hv_weights);

                solution_out[j] = solution_in[j] + dt* ( feval1 );
/*
                // reset boundary solution
                this->enforceBoundaryConditions(s, cur_time);
*/
        }
}
