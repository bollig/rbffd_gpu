//#include "useDouble.cl"
#include "useDouble.cl"

__kernel void
advanceEuler(
        __global FLOAT* solution_in,
        uint offset_to_set,
        uint nb_stencils,
        uint nb_nodes,
        uint stencil_size,
        float dt,
        float cur_time,
        __global FLOAT* solution_out
        )
{
        uint i = get_global_id(0);
        uint j = i + offset_to_set;

        if(i < nb_stencils) {
                // This routine will apply our weights to "s" in however many intermediate steps are required
                // and store the results in feval1
                FLOAT feval1 = solve(solution_in, nb_stencils, nb_nodes, cur_time);

                // compute u^* = u^n + dt*lapl(u^n)
                for (unsigned int i = 0; i < nb_nodes; i++) {
                    solution_out[i] = solution_in[i] + dt* ( feval1 );
               }
/*
                // reset boundary solution
                this->enforceBoundaryConditions(s, cur_time);
*/
        }
}
