__kernel void           
advanceEuler(
             __global uint* stencils,    
             __global FLOAT* lapl_weights,   
             __global FLOAT* x_weights,   
             __global FLOAT* y_weights,   
             __global FLOAT* z_weights,   
             __global FLOAT* solution_in,  
             __global FLOAT* deriv_solution_in,  
             __global FLOAT* diffusivity,  
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
__global uint* st = stencils + j*stencil_size;
__global FLOAT* lapl_st_weights = lapl_weights + j*stencil_size; 
__global FLOAT* x_st_weights = x_weights + j*stencil_size;  
__global FLOAT* y_st_weights = y_weights + j*stencil_size;  
__global FLOAT* z_st_weights = z_weights + j*stencil_size;  

// FIXME: add support for diffusion based on node position
float4 node = (float4)(0.f, 0.f, 0.f, 0.f);
//  = node_list[i];

diffusivity[j] = getDiffusionCoefficient(node, solution_in[j], cur_time, diffusivity[j]);
// Solve for each laplacian using the rewritten form (RHS): 
//          div(k.grad(u) = grad(k).grad(u) + k . lapl(u)
                          // NOTE: the lhs requires interprocessor communication for grad(u)
                          // before computing div(...)
                          FLOAT lapl_u = rewrittenLaplacian(st, lapl_st_weights, x_st_weights, y_st_weights, z_st_weights, deriv_solution_in, diffusivity, nb_stencils, nb_nodes, stencil_size, j);  

                          //To apply weights for a deriv:  applyWeights1PerThread(st, st_weights, solution_in, stencil_size);

                          // FIXME: no forcing yet
                          FLOAT f = 0.f;
                          solution_out[j] = solution_in[j] + dt * (lapl_u + f); 
                          }
                          }

