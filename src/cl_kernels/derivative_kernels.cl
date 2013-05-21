#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#include "computeDeriv.cl"
std::string kernel_source = computeDeriv_source + STRINGIFY_WITH_SUBS(

// Kernel wrapper for computeDerivFLOAT (allow CPU access to the routine)
__kernel void computeDerivKernel(       \n
         __global uint* stencils,    \n
         __global FLOAT* weights,   \n
         __global FLOAT* solution,  \n
         __global FLOAT* derivative,    \n
   uint nb_stencils, \n
   uint stencil_size)  \n
{   \n
	for (int i=0; i < 10; i++) { \n
    computeDeriv(stencils, weights, solution, derivative, nb_stencils, stencil_size); \n
	} \n
}
//----------------------------------------------------------

);
