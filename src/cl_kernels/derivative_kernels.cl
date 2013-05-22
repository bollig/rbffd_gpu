//#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
//#define STRINGIFY(s) #s

#include "useDouble.cl"
#include "computeDeriv.cl"
#include "computeDerivMulti.cl"
//std::string kernel_source = computeDeriv_source + STRINGIFY_WITH_SUBS(

// Kernel wrapper for computeDerivFLOAT (allow CPU access to the routine)
__kernel void computeDerivKernel(       
         __global uint* stencils,    
         __global FLOAT* weights,   
         __global FLOAT* solution,  
         __global FLOAT* derivative,    
   uint nb_stencils, 
   uint stencil_size)  
{   
	for (int i=0; i < 10; i++) { 
    computeDeriv(stencils, weights, solution, derivative, nb_stencils, stencil_size); 
	} 
}
//----------------------------------------------------------
// GPU Only routine
__kernel void computeDerivMultiKernel(       
         __global uint* stencils,     // double4
         __global FLOAT* wx,    // double4
         __global FLOAT* wy,    // double4
         __global FLOAT* wz,    // double4
         __global FLOAT* wl,    // double4
         __global FLOAT* solution,   // (has n FLOATS)
         __global FLOAT* derx,     // double4
         __global FLOAT* dery,     // double4
         __global FLOAT* derz,     // double4
         __global FLOAT* derl,     // double4
   uint nb_stencils, 
   uint stencil_size)  
{   
	for (int i=0; i < 10; i++) { 
    	computeDerivMulti(stencils, 
			wx, wy, wz, wl,
			solution, 
			derx, dery, derz, derl, 
			nb_stencils,
			stencil_size);
	} 
}

//);
