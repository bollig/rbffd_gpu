#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#include "computeDeriv.cl"
std::string kernel_source = computeDeriv + STRINGIFY_WITH_SUBS(

        // Kernel wrapper for computeDerivFLOAT (allow CPU access to the routine)
__kernel void computeDerivKernel(       \n
         __global int* stencils,    \n
         __global FLOAT* weights,   \n
         __global FLOAT* solution,  \n
         __global FLOAT* derivative,    \n
   int nb_stencils, \n
   int stencil_size)  \n
{   \n
    computeDeriv(stencils, weights, solution, derivative, nb_stencils, stencil_size);
}

);
