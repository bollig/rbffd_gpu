#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#include "computeDeriv.cl"
std::string laplacian_source = computeDeriv_source + STRINGIFY_WITH_SUBS(
       
FLOAT rewrittenLaplacian(\n
        // NOTE: this is a SINGLE stencil\n
         __global uint* stencil,    \n
         // NOTE: these are st_weights (i.e., weights for ONE stencil)\n
         // and of stencil_size length. \n
         __global FLOAT* lapl_st_weights,   \n
         __global FLOAT* x_st_weights,   \n
         __global FLOAT* y_st_weights,   \n
         __global FLOAT* z_st_weights,   \n
         // u_t and diffusion are nb_nodes length. \n
         __global FLOAT* u_t,  \n
         __global FLOAT* diffusion,  \n
                   uint nb_stencils, \n
                   uint nb_nodes, \n
                   uint stencil_size, \n
                   uint st_indx)\n
{\n
    FLOAT u_lapl_deriv = applyWeights1PerThread(stencil, lapl_st_weights, u_t, stencil_size); \n
    FLOAT K_dot_lapl_U = diffusion[st_indx] * u_lapl_deriv; \n
\n
    // If we have non-uniform diffusion, more derivatives are requried\n
    FLOAT u_x_deriv = applyWeights1PerThread(stencil, x_st_weights, u_t, stencil_size); \n
    FLOAT u_y_deriv = applyWeights1PerThread(stencil, y_st_weights, u_t, stencil_size); \n
    FLOAT u_z_deriv = applyWeights1PerThread(stencil, z_st_weights, u_t, stencil_size); \n
\n
    FLOAT diff_x_deriv = applyWeights1PerThread(stencil, x_st_weights, diffusion, stencil_size); \n
    FLOAT diff_y_deriv = applyWeights1PerThread(stencil, y_st_weights, diffusion, stencil_size); \n
    FLOAT diff_z_deriv = applyWeights1PerThread(stencil, z_st_weights, diffusion, stencil_size); \n
\n
    FLOAT grad_K_dot_grad_U = diff_x_deriv * u_x_deriv + diff_y_deriv * u_y_deriv + diff_z_deriv * u_z_deriv; 
\n
   return grad_K_dot_grad_U + K_dot_lapl_U;\n
}



        );
