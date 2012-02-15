#include "useDouble.cl"

// Compare: 
//  1) Stencil unpadded and loop over all elements
//  2) Stencil padded but loop ends at stencil_size
//  3) Stencil padded and loop extends to stencil_padded_size (assume 0 weights, and index points to stencil center. The 0s wont cause NaNs)
//  4) For padded cases we need to try BOTH multiple of 16 and 32 padding. 16 because thats the number of DOUBLES in a memop; 32 because its the warp size. 
//  5) For all cases we need to test on LARGE data sizes. We should be able to use the Sphere CVT code or the datasets that I already have. Also, the stencils will be large epsilon which will produce bad derivatives, but will avoid NaNs due to illconditioning in the stencil weight calculations.


FLOAT applyWeights(__global FLOAT* weights, __global FLOAT* u, unsigned int indx, __global uint* stencils, uint stencil_size, uint stencil_padded_size)
{
        // This __global will change to __constant if we can fit the weight into const memory (OPTIMIZATION TODO)
    __global uint* stencil = stencils + indx * stencil_padded_size;
    __global FLOAT* st_weights = weights + indx * stencil_padded_size;

    FLOAT der = 0.0f;       
    uint stencil_end = stencil_size; 
    for (uint j = 0; j < stencil_end; j++) {        
        der += u[stencil[j]] * st_weights[j];
    }   
    return der; 
}


void applyWeights_block(__global FLOAT* weights, __global FLOAT* u, unsigned int indx, __global uint* stencils, uint stencil_size, uint stencil_padded_size, __local FLOAT* der_buf)
{
    __global uint* stencil = stencils + indx * stencil_padded_size;
    __global FLOAT* st_weights = weights + indx * stencil_padded_size;

    // The number of banks in shared memory on the FERMI
    uint num_banks = 1;

    uint lid = get_local_id(0); 
    uint block_size = get_local_size(0);

    uint stencil_end = stencil_size; 

    der_buf[lid*num_banks] = 0.0;

    uint i = 0; 
    uint count = 0; 
    // Repeat process until all weights are applied by block
    while (i < stencil_end) {
        uint j = count*block_size + lid; 
        // Assuming we are under the stencil size, add combination to shared buffer
        if (j < stencil_end) {
            der_buf[lid*num_banks] += u[stencil[j]] * st_weights[j];
        }
        count++;
        i += block_size; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        // (FIXME): change this to a prefix sum
        for (uint i = 1; i < block_size; i++) {
           der_buf[0] += der_buf[i*num_banks]; 
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

