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


void applyWeights_block(__global FLOAT* weights, __global FLOAT* u, unsigned int indx, __global uint* stencils, uint stencil_size, __local FLOAT* der_buf)
{
    __global uint* stencil = stencils + indx * stencil_size;
    __global FLOAT* st_weights = weights + indx * stencil_size;

    uint lid = get_local_id(0); 
    uint block_size = get_local_size(0);

    der_buf[lid*2] = 0.0;

    uint i = 0; 
    uint count = 0; 
    // Repeat process until all weights are applied by block
    while (i < stencil_size) {
        uint j = count*block_size + lid; 
        // (TODO): optimize this random access.
        //      Strategies: 
        //          a) (Avoid __global:) load u as an image so we have better caching
        //          b) (Avoid conds here:) align stencils in memory so they index a u[N+1]==0 and st_weights[n+1] == 0
        //          
        FLOAT uval = (j < stencil_size) ? u[stencil[j]] : 0.;
        FLOAT weight = (j < stencil_size) ? st_weights[j] : 0.;
        // Assuming we are under the stencil size, add combination to shared buffer
        der_buf[lid] += uval * weight;
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
}

