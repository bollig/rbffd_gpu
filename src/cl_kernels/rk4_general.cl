#include "useDouble.cl"

typedef struct Params {
    uint nb_stencils;           
    uint nb_nodes;              
    uint stencil_size;          
    // TODO: add weights pointers
} Params;
/*
__kernel void           
advanceRK4( 
         __global uint* stencils,               
         __global FLOAT* solution_in,           
         __global FLOAT* solution_out,          
                  float dt,                     
                  float cur_time,               
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

        double* k1 = solve(stencils, weights, solution_in, nb_stencils, nb_nodes, stencil_size, dt, cur_time); 

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
        
        double k2 = solve(stencils, weights, k2_input, nb_stencils, nb_nodes, stencil_size, dt, cur_time+0.5*dt); 
        
        // barrier

        k3_input = solution_in + 0.5 * k2; 

        double k3 = solve(stencils, weights, k3_input, nb_stencils, nb_nodes, stencil_size, dt, cur_time+0.5*dt); 
        
        // barrier

        k4_input = solution_in + k3; 

        double k4 = solve(stencils, weights, k4_input, nb_stencils, nb_nodes, stencil_size, dt, cur_time+dt); 
  
        // The k's have dt rolled into them
        solution_out[i] = solution_in[i] + (k1 + 2.*k2 + 2.*k3 + k4)/6.; 
    }
}
*/

// -----------------------------------------------------------
//  Evaluate substep 1, 2 or 3 and scale the output by "k_scale"
//  NOTE: this also adds u_in to k1,k2 or k3, so we can avoid 
//        calling for a in2 = saxpy(1, u, 0.5, k1) before passing
//        in2 to the next substep. This means we avoid a full 
//        kernel call, BUT we need to change the RK4 logic on
//        the last step (below).
// -----------------------------------------------------------

__kernel void           
evaluateRK4_substep( 
         __global FLOAT* u_plus_scaled_k_in,           
         __global FLOAT* u_plus_scaled_k_out,          
                  float k_scale,         
                  float dt,                     
                  float cur_time,               
         __global uint* stencils,    
         __global uint* weights,    
         __constant Params* params 
)  
{   
    uint i = get_global_id(0);    

    if(i < params->nb_stencils) {    
        double feval = 1.0; //solve(u_plus_scaled_k_in, nb_stencils, dt, cur_time, params); 
        u_plus_scaled_k_out[i] = u_plus_scaled_k_in[i] + k_scale * feval; 
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
                  float dt,                     
                  float cur_time,               
         __global uint* stencils,    
         __global uint* weights,    
         __constant Params* param 
)  
{   
    uint i = get_global_id(0);    

    if(i < param->nb_stencils) {    
        double sol = u_in[i]; 

        // Note: k1 and k2 are scaled by 0.5, but we do NOT remove that scale.
        //       Instead we adjust the scalar in the equation below
        double k1 = u_plus_scaled_k1[i] - sol; 
        double k2 = u_plus_scaled_k2[i] - sol; 
        double k3 = u_plus_scaled_k3[i] - sol; 
        // Solve for k4
        double k4 = 1.;//solve(stencils, weights, sol_plus_k3_in, nb_stencils, nb_nodes, stencil_size, dt, cur_time + dt); 

        // See logic above. 
        u_out[i] = sol + (2.*k1 + 4.*k2 + 2.*k3 + k4)/6.; 
    }
}
