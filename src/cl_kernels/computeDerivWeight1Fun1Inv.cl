
#include "useDouble.cl"

// Weights: [stencil_point][rbf_node]
//           [slowest][...][fastest]
void computeDerivWeight1Fun1Inv(
         __global int* stencils,  
         __global FLOAT* ww,     
         __global FLOAT* solution, 
         __global FLOAT* derx,  
   		int nb_stencils, 
   		int stencil_size)  
{   
   int i = get_global_id(0);    

   if (i < nb_stencils) {    
        FLOAT dx1 = 0.0;       

		// point "i" handled by a single thread
		// solution is "reused", but is there a guarantee it will remain in cache? 
		// would it be possible to assign a different thread to each derivative? 
        for (int j = 0; j < stencil_size; j++) {        
            //uint indx = i*stencil_size + j; // stencil[rbf_node][stencil_point]
            int indx = j*nb_stencils + i;   //   stencil[stencil_point][rbf_node]
			int sten = stencils[indx];
			FLOAT s1 = solution[sten];
			FLOAT w0 = ww[indx];
            dx1 += s1 * w0;
        }   

        derx[i] = dx1;    
   }    
}
//----------------------------------------------------------------------
// Weights: [rbf_node][stencil_point]
//           [slowest][...][fastest]
void computeDerivWeight1Fun1(
         __global int* stencils,  
         __global FLOAT* ww,     
         __global FLOAT* solution, 
         __global FLOAT* derx,  
   		int nb_stencils, 
   		int stencil_size)  
{   
   int i = get_global_id(0);    

   if (i < nb_stencils) {    
        FLOAT dx1 = 0.0;       

		// point "i" handled by a single thread
		// solution is "reused", but is there a guarantee it will remain in cache? 
		// would it be possible to assign a different thread to each derivative? 
        for (int j = 0; j < stencil_size; j++) {        
            uint indx = i*stencil_size + j;   // stencil[rbf_node][stencil_point]
            //int indx = j*nb_stencils + i;   //   stencil[stencil_point][rbf_node]
			int sten = stencils[indx];
			FLOAT s1 = solution[sten];
			FLOAT w0 = ww[indx];
            dx1 += s1 * w0;
        }   

        derx[i] = dx1;    
   }    
}
//----------------------------------------------------------------------
#if 0
// Weights: [rbf_node][stencil_point]
//           [slowest][...][fastest]
void computeDeriv4Weight1Fun1(
         __global int4* stencils,  
         __global double4* ww,     
         __global double4* solution, 
         __global double4* derx,  
   		int nb_stencils, 
   		int stencil_size)  
{   
   int i = get_global_id(0);    

   // nb_stencils must be divible by 4
   if (i < (nb_stencils>>2)) {  
        double4 dx1 = 0.0;       

		// point "i" handled by a single thread
		// solution is "reused", but is there a guarantee it will remain in cache? 
		// would it be possible to assign a different thread to each derivative? 
        for (int j = 0; j < stencil_size; j++) {        
            int indx = i*stencil_size + j;   // stencil[rbf_node][stencil_point]
            //int indx = j*nb_stencils + i;   //   stencil[stencil_point][rbf_node]
			int4 sten = stencils[indx];
			double4 s1 = solution[sten];
			double4 w0 = ww[indx];
            dx1 += s1 * w0;
        }   

        derx[i] = dx1;    
   }
}
#endif
//----------------------------------------------------------------------
