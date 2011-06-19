#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#include "laplacian.cl"
std::string kernel_source = laplacian_source + STRINGIFY_WITH_SUBS(

__kernel void           \n
evaluateStep(       \n
//         __global float4* node_list,\n
         __global uint* stencils,    \n
         __global FLOAT* lapl_weights,   \n
         __global FLOAT* x_weights,   \n
         __global FLOAT* y_weights,   \n
         __global FLOAT* z_weights,   \n
         __global FLOAT* solution_in,  \n
         __global FLOAT* deriv_solution_in,  \n
         __global FLOAT* diffusivity,  \n
                   uint offset_to_set, \n
                   uint nb_stencils, \n
                   uint nb_nodes, \n
                   uint stencil_size, \n
                   float dt, \n
                   float cur_time, \n
        __global FLOAT* solution_out\n
)  \n
{   \n
    uint i = get_global_id(0);    \n
    uint j = i + offset_to_set;
        \n
    if(i < nb_stencils) {    \n
        __global uint* st = stencils + j*stencil_size;\n
        __global FLOAT* lapl_st_weights = lapl_weights + j*stencil_size; \n
        __global FLOAT* x_st_weights = x_weights + j*stencil_size;  
        __global FLOAT* y_st_weights = y_weights + j*stencil_size;  
        __global FLOAT* z_st_weights = z_weights + j*stencil_size;  
    \n
        // FIXME: add support for diffusion based on node position\n
        float4 node = (float4)(0.f, 0.f, 0.f, 0.f);\n
        //  = node_list[i];\n
       \n
        diffusivity[j] = getDiffusionCoefficient(node, solution_in[j], cur_time, diffusivity[j]);\n
        // Solve for each laplacian using the rewritten form (RHS): \n
        //          div(k.grad(u) = grad(k).grad(u) + k . lapl(u)\n
        // NOTE: the lhs requires interprocessor communication for grad(u)\n
        // before computing div(...)\n
        FLOAT lapl_u = rewrittenLaplacian(st, lapl_st_weights, x_st_weights, y_st_weights, z_st_weights, deriv_solution_in, diffusivity, nb_stencils, nb_nodes, stencil_size, j);  \n
\n
       //To apply weights for a deriv:  applyWeights1PerThread(st, st_weights, solution_in, stencil_size);\n
\n
        // FIXME: no forcing yet\n
        FLOAT f = 0.f;\n
        solution_out[j] = solution_in[j] + dt * (lapl_u + f); \n
    }\n
}\n

);
