#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#include "computeDeriv.cl"
std::string kernel_source = STRINGIFY_WITH_SUBS(

__kernel void           
advanceFirstOrderEuler(       
         __global int* stencils,    
         __global FLOAT* lapl_weights,   
         __global FLOAT* solution_in,  
                   int nb_stencils, 
                   int nb_nodes, 
                   int stencil_size, 
        __global FLOAT* solution_out
)  
{   
    size_t i = get_global_id(0);    \n

    if(i < nb_stencils) {    \n

//        lapl_u = applyWeights1PerThread(stencils[i*stencil_size], lapl_weights[i*stencil_size], solution);

        // FIXME: no forcing yet
        float f = 0.f;
        solution_out[i] = solution_in[i]; //+ dt * (lapl_u + f); 
    }
}

);
