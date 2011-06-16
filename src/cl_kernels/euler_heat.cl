#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#include "laplacian.cl"
std::string kernel_source = laplacian_source + STRINGIFY_WITH_SUBS(

__kernel void           \n
advanceFirstOrderEuler(       \n
         __global float3* node_list,
         __global int* stencils,    \n
         __global FLOAT* lapl_weights,   \n
         __global FLOAT* x_weights,   \n
         __global FLOAT* y_weights,   \n
         __global FLOAT* z_weights,   \n
         __global FLOAT* solution_in,  \n
         __global FLOAT* diffusivity,  \n
                   int nb_stencils, \n
                   int nb_nodes, \n
                   int stencil_size, \n
                   FLOAT dt, \n
                   FLOAT cur_time, \n
        __global FLOAT* solution_out\n
)  \n
{   \n
    size_t i = get_global_id(0);    \n\n
\n
    if(i < nb_stencils) {    \n\n
        __global int* st = stencils + i*stencil_size;\n
        __global FLOAT* st_weights = lapl_weights + i*stencil_size; \n
    
        float3 node = node_list[i];
       
        diffusivity[i] = getDiffusionCoefficient(node, solution_in[i], cur_time, diffusivity[i]);

        // Solve for each laplacian using the rewritten form (RHS): 
        //          div(k.grad(u) = grad(k).grad(u) + k . lapl(u)
        // NOTE: the lhs requires interprocessor communication for grad(u)
        // before computing div(...)
        FLOAT lapl_u = rewrittenLaplacian(st, lapl_weights, x_weights, y_weights, z_weights, solution_in, diffusivity, nb_stencils, nb_nodes, stencil_size, i);  

       //To apply weights for a deriv:  applyWeights1PerThread(st, st_weights, solution_in, stencil_size);\n
\n
        // FIXME: no forcing yet\n
        FLOAT f = 0.f;\n
        solution_out[i] = solution_in[i] + dt * (lapl_u + f); \n
    }\n
}\n

);
