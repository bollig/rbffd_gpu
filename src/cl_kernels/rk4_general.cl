// BAD : this doesnt work. Use rk4_classic or another rk4 instead


#include "useDouble.cl"
//Assume the "solver" will include this file instead of the other way around
//#include "solver.cl"

// -----------------------------------------------------------
//  Evaluate substep 1, 2 or 3 and scale the output by "k_scale"
//  NOTE: this also adds u_in to k1,k2 or k3, so we can avoid 
//        calling for a in2 = saxpy(1, u, 0.5, k1) before passing
//        in2 to the next substep. This means we avoid a full 
//        kernel call, BUT we need to change the RK4 logic on
//        the last step (below).
// -----------------------------------------------------------

__kernel
void evaluateRK4_substep(
    __global FLOAT* u_plus_scaled_k_in,
    __global FLOAT* u_plus_scaled_k_out,
    double dt,
    double cur_time,
    double k_scale,

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
            // --------------
            //   u+0.5*dt*feval1

            // In 1: u, t                       Out 1: u+0.5*(f1)
            // In 2: u+0.5*(f1), t+0.5*dt       Out 2: u+0.5*(f2)
            // In 3: u+0.5*(f2), t+0.5*dt       Out 3: u + 1*(f3)
            // In 4: u,t, u+0.5(f1), u+0.5(f2), u+1*(f3)        

            //  Out 4: u + (dt/6) * (k1 + 2k2 + 2k3 + k4) 
            //          which means we need: 
            //         u + (dt/6) * (2(k1 - u) + 4(k2-u) + 2(k3-u) + k4)
            
        u_plus_scaled_k_out[j] = u_plus_scaled_k_in[j] + k_scale * dt * feval1;
    }
}




// -----------------------------------------------------------
// Evaluate the final substep of RK4 and advance the solution
// -----------------------------------------------------------
__kernel void           
advanceRK4_substeps( 
         __global FLOAT* u_in,           
         __global FLOAT* u_out,
         __global FLOAT* u_plus_scaled_k1,          
         __global FLOAT* u_plus_scaled_k2,          
         __global FLOAT* u_plus_scaled_k3,          
    double dt,
    double cur_time,

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
        FLOAT sol = u_in[j];

        // Note: k1 and k2 are scaled by 0.5, but we do NOT remove that scale.
        //       Instead we adjust the scalar in the equation below
        FLOAT k1 = u_plus_scaled_k1[j] - sol;
        FLOAT k2 = u_plus_scaled_k2[j] - sol;
        FLOAT k3 = u_plus_scaled_k3[j] - sol;
        // Solve for k4
        FLOAT k4 = solve(u_plus_scaled_k3,
                             j,
                             cur_time+dt,
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

        // See logic above. 
        u_out[j] = sol + (2.*k1 + 4.*k2 + 2.*k3 + dt*k4) / 6.;
    }
}



/*

   __kernel void           
advanceRK4( 
         __global uint* stencils,               
         __global FLOAT* solution_in,           
         __global FLOAT* solution_out,          
                  double dt,                     
                  double cur_time,               
         __constant Params* param 
)  
{   
    uint i = get_global_id(0);    

    if(i < param->nb_stencils) {    
        __global uint* st = stencils + i* param->stencil_size;

        //    k1 = dt*func(DM_Lambda, DM_Theta, H, u, t, nodes, useHV);
        //    k2 = dt*func(DM_Lambda, DM_Theta, H, u+0.5*F1, t+0.5*dt, nodes, useHV);
        //    k3 = dt*func(DM_Lambda, DM_Theta, H, u+0.5*F2, t+0.5*dt, nodes, useHV);
        //    k4 = dt*func(DM_Lambda, DM_Theta, H, u+F3, t+dt, nodes, useHV);

        FLOAT* k1 = solve(stencils, weights, solution_in, nb_stencils, nb_nodes, stencil_size, dt, cur_time); 

        // global barrier required. probably best if we do eval+saxpy kernel combo and specify: 
        //      kernel(solution_in, solution_in, 0, k1_out)  =>      calls solve(stencils, weights, solution_in + 0*solution_in, nb_stencils, nb_nodes, stencil_size, dt, cur_time + 0*dt)       // Matches k1
        //      kernel(solution_in, k1_out, 0.5, k2_out)     =>      calls solve(stencils, weights, solution_in + 0.5*k1_out, nb_stencils, nb_nodes, stencil_size, dt, cur_time + 0.5*dt)        // Matches k2
        //      kernel(solution_in, k2_out, 0.5, k3_out)     =>      calls solve(stencils, weights, solution_in + 0.5*k2_out, nb_stencils, nb_nodes, stencil_size, dt, cur_time + 0.5*dt)        // Matches k3
        //      kernel(solution_in, k3_out, 1.0, k4_out)     =>      calls solve(stencils, weights, solution_in + 1.0*k4_out, nb_stencils, nb_nodes, stencil_size, dt, cur_time + 1.0*dt)        // Matches k3
        // problem: saxpy on parameter passed to solve()
        // solution: assume kernel call outputs "solution_in_x + scale * solution_in_y"
        // modification: the final kernel will need to sum: 
        //   solution_in[i] + (k1 + 2.*k2 + 2.*k3 + k4)/6.
        // but, because solution_in is added to each k, we will need to subtract it as preprocess
        // This will avoid a barrier; since solution_in is already loaded for the first term of the sum, 
        // we only add arithmetic and not memloads. 

        // Run loop for stencil: 
        k2_input = solution_in + 0.5 * k1
        
        FLOAT k2 = solve(stencils, weights, k2_input, nb_stencils, nb_nodes, stencil_size, dt, cur_time+0.5*dt); 
        
        // barrier

        k3_input = solution_in + 0.5 * k2; 

        FLOAT k3 = solve(stencils, weights, k3_input, nb_stencils, nb_nodes, stencil_size, dt, cur_time+0.5*dt); 
        
        // barrier

        k4_input = solution_in + k3; 

        FLOAT k4 = solve(stencils, weights, k4_input, nb_stencils, nb_nodes, stencil_size, dt, cur_time+dt); 
  
        // The k's have dt rolled into them
        solution_out[i] = solution_in[i] + (k1 + 2.*k2 + 2.*k3 + k4)/6.; 
    }
}
*/


