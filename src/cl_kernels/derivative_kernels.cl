#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#include "computeDeriv.cl"
#include "computeDerivMulti.cl"
std::string kernel_source = computeDeriv_source + computeDerivMulti_source + STRINGIFY_WITH_SUBS(

// Kernel wrapper for computeDerivFLOAT (allow CPU access to the routine)
__kernel void computeDerivKernel(       \n
         __global uint* stencils,    \n
         __global FLOAT* weights,   \n
         __global FLOAT* solution,  \n
         __global FLOAT* derivative,    \n
   uint nb_stencils, \n
   uint stencil_size)  \n
{   \n
    computeDeriv(stencils, weights, solution, derivative, nb_stencils, stencil_size);
}
//----------------------------------------------------------
// Kernel wrapper for computeDerivMultiFLOAT (allow CPU access to the routine)
__kernel void computeDerivMultiKernel(       \n
         __global uint* stencils,    \n
         __global FLOAT* weights,   \n
         __global FLOAT* solution,  \n
         __global FLOAT* derivative,    \n
   uint nb_stencils, \n
   uint stencil_size)  \n
{   \n
    computeDerivMulti(stencils, weights, solution, derivative, nb_stencils, stencil_size);
}
//----------------------------------------------------------

);
