#include "useDouble.cl"

// Compute 4 derivatives of a single function

void computeDerivMultiWeight(     
         __global int* stencils,     // double4
         __global FLOAT* ww,    // double4 (allocated 4x the space)
         __global FLOAT* solution,   // (has n FLOATS)
         __global FLOAT* derx,     // double4
         __global FLOAT* dery,     // double4
         __global FLOAT* derz,     // double4
         __global FLOAT* derl,     // double4
   int nb_stencils, 
   int stencil_size)  
{   
   int i = get_global_id(0);    

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
        for (int j = 0; j < stencil_size; j++) {        
            int indx = i*stencil_size + j;
			int ind  = indx << 2;
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
// Single function, 4 derivatives, using double4
void computeDeriv4Weight4Fun1(     
         __global int* stencils, 
         __global double4* ww, 
         __global double* solution, 
         __global double4* der,
   		int nb_stencils, 
   		int stencil_size)  
{   
   int i = get_global_id(0);    

   if(i >= nb_stencils) return;

   {
        double4 dx = (double4) (0.,0.,0.,0.);       

        for (int j = 0; j < stencil_size; j++) {        
            int indx = i*stencil_size + j;
			double sol = solution[stencils[indx]];
            dx += sol * ww[indx];      // multiplication occuring correctly. 
        }   
        der[i] = dx;
   }    
}
//----------------------------------------------------------------------
// Single function, 4 derivatives, using double
// derivatives are NOT interleaved
void computeDeriv1Weight4Fun1(     
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
   int i = get_global_id(0);    

   if(i >= nb_stencils) return;

   {
        double dx = 0.;
        double dy = 0.;
        double dz = 0.;
        double dl = 0.;
		//double sol = 1.;

		#if 1
        for (int j = 0; j < stencil_size; j++) { 
            int indx = i*stencil_size + j;
			int ind  = indx << 2;
			#if 1
			double sol = solution[stencils[indx]];
			#else
			#endif


			#if 1
			#if 1     
            dx += sol * ww[ind];     // 125ms with memory update (non-optimized)
            dy += sol * ww[ind+1];   // 140ms when optimized (hard to believe)
            dz += sol * ww[ind+2];    
            dl += sol * ww[ind+3];    
			#else
            dx += sol;    // cost: 88ms with memory update
            dy += sol;   
            dz += sol;
            dl += sol;
			#endif
			#endif
        }   
		#endif

		#if 1
        derx[i] = dx;     // cost: 15ms wihtout rest of code (unoptimized)
        dery[i] = dy;    
        derz[i] = dz;    
        derl[i] = dl;    
		#endif
   }    
   //if (i == 0) printf("exit computeDeriv1Weight4Fun1 CL function\n");
}
//----------------------------------------------------------------------
// compute 4 derivatives of four functions
// Different formats possible for ww, sol, der
// Two main cases: store (ux,uy,...) close in memory, 
// or store (ux,vx,wx,wl) close in memory. 
// Third case: (store 16 derivatives close in memory). 
// The three cases change how data is input into the function. 

void computeDeriv4Weight4Fun4(     
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
   int i = get_global_id(0);    

   if(i >= nb_stencils) return;

   {
        double4 dd0 = 0.;  // d/d{x,y,z,l}(sol0)
        double4 dd1 = 0.;
        double4 dd2 = 0.;
        double4 dd3 = 0.;
		//double sol = 1.;

		#if 1
        for (int j = 0; j < stencil_size; j++) { 
            int indx = i*stencil_size + j;
			#if 1
			int ind = stencils[indx];
			// The solution vectors are not compact
			double sol0 = solution[ind];
			double sol1 = solution[ind +   nb_stencils];
			double sol2 = solution[ind + 2*nb_stencils];
			double sol3 = solution[ind + 3*nb_stencils];
			#else
			#endif


			#if 1
            dd0 += sol0 * ww[ind];     // 125ms with memory update (non-optimized)
            dd1 += sol1 * ww[ind];     // 125ms with memory update (non-optimized)
            dd2 += sol2 * ww[ind];     // 125ms with memory update (non-optimized)
            dd3 += sol3 * ww[ind];     // 125ms with memory update (non-optimized)
			#endif
        }   
		#endif

		#if 1
        ders0[i] = dd0;     // cost: 15ms wihtout rest of code (unoptimized)
        ders1[i] = dd1;    
        ders2[i] = dd2;    
        ders3[i] = dd3;    
		#endif
   }    
   //if (i == 0) printf("exit computeDeriv1Weight4Fun1 CL function\n");
}
//----------------------------------------------------------------------
