#include "useDouble.cl"
    
FLOAT applyWeights(__global FLOAT* weights, __global FLOAT* u, unsigned int indx, __global uint* stencils, uint stencil_size, uint stencil_padded_size)
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


void applyWeights_block(__global FLOAT* weights, __global FLOAT* u, unsigned int indx, __global uint* stencils, uint stencil_size, uint stencil_padded_size, __local FLOAT* der_buf)
{
    __global uint* stencil = stencils + indx * stencil_size;
    __global FLOAT* st_weights = weights + indx * stencil_size;

    uint lid = get_local_id(0); 
    uint block_size = get_local_size(0);

    der_buf[lid] = 0.0;

    uint i = 0; 
    uint count = 0; 
    // Repeat process until all weights are applied by block
    while (i < stencil_size) {
        uint j = count*block_size + lid; 
        // Assuming we are under the stencil size, add combination to shared buffer
        if (j < stencil_size) {
            der_buf[lid] += u[stencil[j]] * st_weights[j];
        }
        count++;
        i += block_size; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        // (FIXME): change this to a prefix sum
        for (uint i = 1; i < block_size; i++) {
           der_buf[lid] += der_buf[i]; 
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

