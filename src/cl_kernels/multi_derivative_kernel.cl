#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#include "computeDerivMulti.cl"
std::string kernel_source = computeDerivMulti_source + STRINGIFY_WITH_SUBS(

//----------------------------------------------------------
// Kernel wrapper for computeDerivMultiFLOAT (allow CPU access to the routine)
// stencils: one stencil per node
// weights: interleaved with weights for 4 dependent variables
// solution: interleaved solution vector with 4 dependent variables
// derv1 interleaved x,y,z,l-derivatives for variable 1
// derv2 interleaved x,y,z,l-derivatives for variable 2
// derv3 interleaved x,y,z,l-derivatives for variable 3
// derv4 interleaved x,y,z,l-derivatives for variable 4
__kernel void computeDerivMultiKernel(       \n
         __global uint* stencils,    \n
         __global FLOAT* wx,   \n
         __global FLOAT* wy,   \n
         __global FLOAT* wz,   \n
         __global FLOAT* wl,   \n
         __global FLOAT* solution,  \n
         __global FLOAT* derx,    \n
         __global FLOAT* dery,    \n
         __global FLOAT* derz,    \n
         __global FLOAT* derl,    \n
   uint nb_stencils, \n
   uint stencil_size)  \n
{   \n
	for (int i=0; i < 10; i++) { \n
    	computeDerivMulti(stencils, wx, wy, wz, wl, solution, derx, dery, derz, derl, nb_stencils, stencil_size); \n
	} \n
}
//----------------------------------------------------------
// Kernel wrapper for computeDerivMulti4FLOAT (allow CPU access to the routine)
// stencils: one stencil per node
// weights: interleaved with weights for 4 dependent variables
// solution: interleaved solution vector with 4 dependent variables
// derv1 interleaved x,y,z,l-derivatives for variable 1
// derv2 interleaved x,y,z,l-derivatives for variable 2
// derv3 interleaved x,y,z,l-derivatives for variable 3
// derv4 interleaved x,y,z,l-derivatives for variable 4
// compute derivate of four functions
__kernel void computeDerivMulti4Kernel(       \n
         __global uint* stencils,    \n
         __global FLOAT* wx,   \n
         __global FLOAT* wy,   \n
         __global FLOAT* wz,   \n
         __global FLOAT* wl,   \n
         __global double4* solution,  \n
         __global FLOAT* derx,    \n
         __global FLOAT* dery,    \n
         __global FLOAT* derz,    \n
         __global FLOAT* derl,    \n
   uint nb_stencils, \n
   uint stencil_size)  \n
{   \n
	for (int i=0; i < 10; i++) { \n
    	computeDerivMulti4(stencils, wx, wy, wz, wl, solution, derx, dery, derz, derl, nb_stencils, stencil_size); \n
	} \n
}
//----------------------------------------------------------

);
