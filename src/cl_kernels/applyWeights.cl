#include "useDouble.cl"
    
FLOAT applyWeights(__global FLOAT* weights, __global FLOAT* u, unsigned int indx, __global uint* stencils, uint stencil_size)
{
        // This __global will change to __constant if we can fit the weight into const memory (OPTIMIZATION TODO)
    __global uint* stencil = stencils + indx * stencil_size;
    __global FLOAT* st_weights = weights + indx * stencil_size;

    FLOAT der = 0.0f;       
    for (uint j = 0; j < stencil_size; j++) {        
        der += u[stencil[j]] * st_weights[j];
    }   
    return der; 
}

