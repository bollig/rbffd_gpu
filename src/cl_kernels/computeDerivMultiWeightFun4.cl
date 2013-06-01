#include "useDouble.cl"

void computeDerivMultiWeightFun4(     
         __global int* stencils,  // double4
         __global FLOAT* ww,       // multiple weights
         __global double4* solution, // multiple functions
         __global double4* derx,     //   "
         __global double4* dery,     //   "
         __global double4* derz,     //   "
         __global double4* derl,     //   "
   int nb_stencils, 
   int stencil_size)  
{   
   int i = get_global_id(0);    

   //double4 xxx = (0.,0.,0.,0.); // VALID EXPRESSION
   // put solution into double4; have single thread work with double4
   // Best to have one thread per stencil value and have them access individual elements of stencil
    
   if(i < nb_stencils) {    

// USE_DOUBLES
#if 0
        ;
// USE_DOUBLES
#else
// USE_DOUBLES
        double4 dx = (0.0,0.,0.,0.);       
        double4 dy = (0.0,0.,0.,0.);       
        double4 dz = (0.0,0.,0.,0.);       
        double4 dl = (0.0,0.,0.,0.);       

        for (int j = 0; j < stencil_size; j++) {        
            int indx = i*stencil_size + j;
			int ind  = indx << 2;
		// 4 weights ==> 32 bytes (wx,wy,wz,wl) at a point
		// 4 functions ==> 32 bytes
		// 4 derivatives ==> 32 bytes
			//int sten = stencils[indx];
			double4 s = solution[stencils[indx]];

			FLOAT w0 = ww[ind];
			FLOAT w1 = ww[ind+1];
			FLOAT w2 = ww[ind+2];
			FLOAT w3 = ww[ind+3];

			dx += s*w0;
			dy += s*w1;
			dz += s*w2;
			dl += s*w3;
		}
		derx[i] = dx;
		dery[i] = dy;
		derz[i] = dz;
		derl[i] = dl;   // lapl of all four solutions
#endif
   }    
}
//----------------------------------------------------------------------
