
#include "useDouble.cl"
#include "computeDeriv.cl"
#include "computeDerivMulti.cl"
#include "computeDerivMultiWeight.cl"

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
         __global FLOAT* restrict wx,    // double4
         __global FLOAT* restrict wy,    // double4
         __global FLOAT* restrict wz,    // double4
         __global FLOAT* restrict wl,    // double4
         __global FLOAT* restrict solution,   // (has n FLOATS)
         __global FLOAT* restrict derx,     // double4
         __global FLOAT* restrict dery,     // double4
         __global FLOAT* restrict derz,     // double4
         __global FLOAT* restrict derl,     // double4
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
//----------------------------------------------------------------------
// GPU Only routine, consolidate weights
__kernel void computeDerivMultiWeightKernel(       
         __global uint* stencils,     // double4
         __global FLOAT* restrict ww,    // double4
         //__global FLOAT* restrict wy,    // double4
         //__global FLOAT* restrict wz,    // double4
         //__global FLOAT* restrict wl,    // double4
         __global FLOAT* restrict solution,   // (has n FLOATS)
         __global FLOAT* restrict derx,     // double4
         __global FLOAT* restrict dery,     // double4
         __global FLOAT* restrict derz,     // double4
         __global FLOAT* restrict derl,     // double4
   uint nb_stencils, 
   uint stencil_size)  
{   
	for (int i=0; i < 10; i++) { 
    	computeDerivMultiWeight(stencils, 
			ww, 
			solution, 
			derx, dery, derz, derl, 
			nb_stencils,
			stencil_size);
	} 
}
//----------------------------------------------------------------------
