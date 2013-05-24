#include "useDouble.cl"

void computeDerivMultiWeight(     
         __global uint* stencils,     // double4
         __global FLOAT* ww,    // double4 (allocated 4x the space)
         //__global FLOAT* wy,    // double4
         //__global FLOAT* wz,    // double4
         //__global FLOAT* wl,    // double4
         __global FLOAT* solution,   // (has n FLOATS)
         __global FLOAT* derx,     // double4
         __global FLOAT* dery,     // double4
         __global FLOAT* derz,     // double4
         __global FLOAT* derl,     // double4
   uint nb_stencils, 
   uint stencil_size)  
{   
   uint i = get_global_id(0);    

   //double4 xxx = (0.,0.,0.,0.); // VALID EXPRESSION
   // put solution into double4; have single thread work with double4
   // Best to have one thread per stencil value and have them access individual elements of stencil
    
   if(i < nb_stencils) {    

        FLOAT dx = 0.0;       
        FLOAT dy = 0.0;       
        FLOAT dz = 0.0;       
        FLOAT dl = 0.0;       
		// point "i" handled by a single thread
		// solution is "reused", but is there a guarantee it will remain in cache? 
		// would it be possible to assign a different thread to each derivative? 
        for (uint j = 0; j < stencil_size; j++) {        
            uint indx = i*stencil_size + j;
			uint ind  = indx << 2;
        //    der += 1. * weights[indx];    
        //    der += 1. ;    
			FLOAT sol = solution[stencils[indx]];
			#if 1
            dx += sol * ww[indx];    
            dy += sol * ww[indx+1];    
            dz += sol * ww[indx+2];    
            dl += sol * ww[indx+3];    
			#else
            dx += sol;
            dy += sol;
            dz += sol;
            dl += sol;
			#endif
        }   
        derx[i] = dx;    
        dery[i] = dy;    
        derz[i] = dz;    
        derl[i] = dl;    
   }    
}
//----------------------------------------------------------------------
