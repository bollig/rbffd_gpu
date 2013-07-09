
#include "useDouble.cl"
#include "computeDeriv.cl"
#include "computeDerivMulti.cl"
#include "computeDerivMultiWeight.cl"
#include "computeDerivMultiWeightFun.cl"
#include "computeDerivWeight1Fun1Inv.cl"
#include "computeDerivMultiWeightFun4.cl"


// Disable vectorization (does not "seem" to have an effect.)

// Kernel wrapper for computeDerivFLOAT (allow CPU access to the routine)
__kernel void computeDerivKernel(       
__global int* stencils,    
__global FLOAT* weights,   
__global FLOAT* solution,  
__global FLOAT* derivative,    
int nb_stencils, 
int stencil_size)  
{   
	for (int i=0; i < 10; i++) { 
    computeDeriv(stencils, weights, solution, derivative, nb_stencils, stencil_size); 
	} 
}
//----------------------------------------------------------
// GPU Only routine
__kernel ATTRIBUTES
void computeDerivMultiKernel(       
         __global int* stencils,     // double4
         __global FLOAT* restrict wx,    // double4
         __global FLOAT* restrict wy,    // double4
         __global FLOAT* restrict wz,    // double4
         __global FLOAT* restrict wl,    // double4
         __global FLOAT* restrict solution,   // (has n FLOATS)
         __global FLOAT* restrict derx,     // double4
         __global FLOAT* restrict dery,     // double4
         __global FLOAT* restrict derz,     // double4
         __global FLOAT* restrict derl,     // double4
   int nb_stencils, 
   int stencil_size)  
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
__kernel ATTRIBUTES
void computeDerivMultiWeightKernel(       
         __global int* stencils,     // double4
         __global FLOAT* restrict ww,    // double4
         //__global FLOAT* restrict wy,    // double4
         //__global FLOAT* restrict wz,    // double4
         //__global FLOAT* restrict wl,    // double4
         __global FLOAT* restrict solution,   // (has n FLOATS)
         __global FLOAT* restrict derx,     // double4
         __global FLOAT* restrict dery,     // double4
         __global FLOAT* restrict derz,     // double4
         __global FLOAT* restrict derl,     // double4
   int nb_stencils, 
   int stencil_size)  
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
// GPU Only routine, consolidate weights
// Weights: [rbf_node][stencil_point][which_deriv]
// works properly
__kernel ATTRIBUTES
void computeDerivMultiWeightFunKernel(       
         __global int* stencils,     // double4
         __global FLOAT* restrict ww4,    // multiple weights
         __global FLOAT* restrict solution,   // multiple functions
         __global FLOAT* restrict derx,     // double4
         __global FLOAT* restrict dery,     // double4
         __global FLOAT* restrict derz,     // double4
         __global FLOAT* restrict derl,     // double4
   int nb_stencils, 
   int stencil_size)  
{   
	for (int i=0; i < 10; i++) {  // number of iterations affecting # HW errors . WHY?
    	computeDerivMultiWeightFun(stencils, 
			ww4, 
			solution, 
			derx, dery, derz, derl, 
			nb_stencils,
			stencil_size);
	} 
}
//----------------------------------------------------------------------
// GPU Only routine, consolidate weights
// Weights: [stencil_point][rbf_node]
// Single weight, single function
__kernel ATTRIBUTES
void computeDerivWeight1Fun1InvKernel(       
         __global int* stencils,   
         __global FLOAT* restrict ww4,    
         __global FLOAT* restrict solution,  
         __global FLOAT* restrict derx,  
   int nb_stencils, 
   int stencil_size)  
{   
	for (int i=0; i < 10; i++) {  // number of iterations affecting # HW errors . WHY?
    	computeDerivWeight1Fun1Inv(stencils, 
			ww4, 
			solution, 
			derx, 
			nb_stencils,
			stencil_size);
	} 
}
//----------------------------------------------------------------------
// GPU Only routine, consolidate weights
// Weights: [stencil_point][rbf_node]
// Single weight, single function
__kernel ATTRIBUTES
void computeDerivWeight1Fun1Kernel(       
         __global int* stencils,   
         __global FLOAT* restrict ww4,    
         __global FLOAT* restrict solution,  
         __global FLOAT* restrict derx,  
   int nb_stencils, 
   int stencil_size)  
{   
	for (int i=0; i < 10; i++) {  // number of iterations affecting # HW errors . WHY?
		//printf("kernel loop, i= %d\n", i);
    	computeDerivWeight1Fun1(stencils, 
			ww4, 
			solution, 
			derx, 
			nb_stencils,
			stencil_size);
	} 
}
//----------------------------------------------------------------------
#if 0
// GPU Only routine, consolidate weights
// Weights: [stencil_point][rbf_node]
// Single weight, single function
// Use double4 instead of double for better efficiency? 
// That means that I must decrease size of item grid by 4
// Deriv4: double4
__kernel void computeDeriv4Weight1Fun1Kernel(       
         __global int4* stencils,   
         __global double4* restrict ww4,    
         __global double4* restrict solution,  
         __global double4* restrict derx,  
   int nb_stencils, 
   int stencil_size)  
{   
	for (int i=0; i < 10; i++) {  // number of iterations affecting # HW errors . WHY?
    	computeDeriv4Weight1Fun1(stencils, 
			ww4, 
			solution, 
			derx, 
			nb_stencils,
			stencil_size);
	} 
}
#endif
//----------------------------------------------------------------------
//----------------------------------------------------------------------
// GPU Only routine, consolidate weights
__kernel __attribute__((vec_type_hint(double4)))
void computeDerivMultiWeightFun4Kernel(       
         __global int* stencils,     // double4
         __global FLOAT* restrict ww4,    // multiple weights
         __global double4* restrict solution,   // multiple functions
         __global double4* restrict derx,     // double4
         __global double4* restrict dery,     // double4
         __global double4* restrict derz,     // double4
         __global double4* restrict derl,     // double4
   		int nb_stencils, 
   		int stencil_size)  
{   
	for (int i=0; i < 10; i++) { 
    	computeDerivMultiWeightFun4(stencils, 
			ww4, 
			solution, 
			derx, dery, derz, derl, 
			nb_stencils,
			stencil_size);
	} 
}
//----------------------------------------------------------------------
__kernel ATTRIBUTES
void computeDeriv4Weight4Fun1Kernel(     
         __global int* stencils, 
         __global double4* ww, 
         __global double* solution, 
         __global double4* der,
   		int nb_stencils, 
   		int stencil_size)  
{   
	for (int i=0; i < 10; i++) { 
		computeDeriv4Weight4Fun1(     
         stencils, 
         ww, 
         solution, 
         der,
   		nb_stencils, 
		stencil_size);
	} 
}
//----------------------------------------------------------------------
__kernel ATTRIBUTES
void computeDeriv1Weight4Fun1Kernel(     
         __global int* stencils, 
         __global double* ww, 
         __global double* solution, 
         __global double* derx,
         __global double* dery,
         __global double* derz,
         __global double* derl,
   		int nb_stencils, 
   		int stencil_size)  
{   
#if 0
	if (get_global_id(0) == 0) {
		printf(">>>> local size: %d\n", get_local_size(0));
		printf(">>>> number workgroups: %d\n", get_num_groups(0));
		printf(">>>> total nb workitems: %d\n", get_global_size(0));
	}
#endif

	for (int i=0; i < 10; i++) { 
		computeDeriv1Weight4Fun1(     
         stencils, 
         ww, 
         solution, 
         derx, dery, derz, derl,
   		nb_stencils, 
		stencil_size);
		//barrier(CLK_GLOBAL_MEM_FENCE);
	} 
}
//----------------------------------------------------------------------
__kernel ATTRIBUTES
void computeDeriv4Weight4Fun4Kernel(     
         __global int* stencils, 
         __global double4* ww, 
         __global double* solution, 
         __global double4* ders0,
         __global double4* ders1,
         __global double4* ders2,
         __global double4* ders3,
   		int nb_stencils, 
   		int stencil_size)  
{
	for (int i=0; i < 10; i++) { 
		computeDeriv4Weight4Fun4(     
         stencils, 
         ww, 
         solution, 
         ders0, ders1, ders2, ders3,
   		nb_stencils, 
		stencil_size);
	} 
}
//----------------------------------------------------------------------
__kernel  ATTRIBUTES
void computeDeriv4Weight4Fun4InvKernel(     
         __global int* stencils, 
         __global double4* ww, 
         __global double* solution, 
         __global double4* ders0,
         __global double4* ders1,
         __global double4* ders2,
         __global double4* ders3,
   		int nb_stencils, 
   		int stencil_size)  
{
	for (int i=0; i < 10; i++) { 
		computeDeriv4Weight4Fun4Inv(     
         stencils, 
         ww, 
         solution, 
         ders0, ders1, ders2, ders3,
   		nb_stencils, 
		stencil_size);
	} 
}
//----------------------------------------------------------------------
