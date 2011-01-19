#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#define EXAMPLE_MACRO 4

std::string kernel_source = STRINGIFY_WITH_SUBS(


__kernel void           \n
computeDerivKernel(             \n
         __global int* stencils,    \n
         __global int* stencil_offset,  \n
         __global float* weights,   \n
         __global float* solution,  \n
         __global float* derivative,    \n
   const unsigned int nb_stencils)  \n
{   \n
   int i = get_global_id(0);    \n
    
   if(i < nb_stencils) {    \n
        int offset = stencil_offset[i];     \n
        int offset_stop = stencil_offset[i+1];  \n 
        int stencil_size = offset_stop - offset;    \n

        // our stencil starts at the offset \n
        __global int* stencil = stencils + stencil_offset[i];    \n
        __global float* weight = weights + stencil_offset[i];    \n

        float der = 0.0f;       \n
        for (int j = 0; j < stencil_size; j++) {        \n
                der += solution[stencil[j]] * weight[j];    \n
        }   \n
        derivative[i] = der;    \n
   }    \n
}

);
