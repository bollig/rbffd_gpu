#include "useDouble.cl"

void computeDerivMultiWeightFun(     
         __global int* stencils,  // double4
         __global FLOAT* ww,       // multiple weights
         __global FLOAT* solution, // multiple functions
         __global FLOAT* derx,     //   "
         __global FLOAT* dery,     //   "
         __global FLOAT* derz,     //   "
         __global FLOAT* derl,     //   "
   int nb_stencils, 
   int stencil_size)  
{   
   int i = get_global_id(0);    

   //double4 xxx = (0.,0.,0.,0.); // VALID EXPRESSION
   // put solution into double4; have single thread work with double4
   // Best to have one thread per stencil value and have them access individual elements of stencil

    
   if(i < nb_stencils) {    

// USE_DOUBLES
#if 1
        FLOAT dx1 = 0.0;       
        FLOAT dy1 = 0.0;       
        FLOAT dz1 = 0.0;       
        FLOAT dl1 = 0.0;       

        FLOAT dx2 = 0.0;       
        FLOAT dy2 = 0.0;       
        FLOAT dz2 = 0.0;       
        FLOAT dl2 = 0.0;       

        FLOAT dx3 = 0.0;       
        FLOAT dy3 = 0.0;       
        FLOAT dz3 = 0.0;       
        FLOAT dl3 = 0.0;       

        FLOAT dx4 = 0.0;       
        FLOAT dy4 = 0.0;       
        FLOAT dz4 = 0.0;       
        FLOAT dl4 = 0.0;       

		// point "i" handled by a single thread
		// solution is "reused", but is there a guarantee it will remain in cache? 
		// would it be possible to assign a different thread to each derivative? 
        for (int j = 0; j < stencil_size; j++) {        
            int indx = i*stencil_size + j;
			int ind  = indx << 2;
        //    der += 1. * weights[indx];    
        //    der += 1. ;    
		// 4 weights ==> 32 bytes (wx,wy,wz,wl) at a point
		// 4 functions ==> 32 bytes
		// 4 derivatives ==> 32 bytes
			int sten = stencils[indx];
			FLOAT s1 = solution[4*sten];
			FLOAT s2 = solution[4*sten+1];
			FLOAT s3 = solution[4*sten+2];
			FLOAT s4 = solution[4*sten+3];

			// REMOVE_WEIGHTS
			#if 1
			FLOAT w0 = ww[ind];
			FLOAT w1 = ww[ind+1];
			FLOAT w2 = ww[ind+2];
			FLOAT w3 = ww[ind+3];

            dx1 += s1 * w0;
            dy1 += s1 * w1;
            dz1 += s1 * w2;
            dl1 += s1 * w3;

            dx2 += s2 * w0;
            dy2 += s2 * w1;
            dz2 += s2 * w2;
            dl2 += s2 * w3;

            dx3 += s3 * w0;
            dy3 += s3 * w1;
            dz3 += s3 * w2;
            dl3 += s3 * w3;

            dx4 += s4 * w0;
            dy4 += s4 * w1;
            dz4 += s4 * w2;
            dl4 += s4 * w3;
			// REMOVE_WEIGHTS
			#else
            dx1 += s1;
            dy1 += s1;
            dz1 += s1;
            dl1 += s1;

            dx2 += s2;
            dy2 += s2;
            dz2 += s2;
            dl2 += s2;

            dx3 += s3;
            dy3 += s3;
            dz3 += s3;
            dl3 += s3;

            dx4 += s4;
            dy4 += s4;
            dz4 += s4;
            dl4 += s4;
			// REMOVE_WEIGHTS
			#endif
        }   
		// derx = (dx1, dx2, dx3, dx4)
		// They should be all equal if weights are (1,0,0,0...)

		int i4 = i << 2;  // multiply by 4
        derx[i4+0] = 1.*dx1;    
        dery[i4+0] = 1.*dx1;    
        derz[i4+0] = 1.*dz1;    
        derl[i4+0] = 1.*dl1;    

        derx[i4+1] = 1.*dx2;    
        dery[i4+1] = dy2;    
        derz[i4+1] = dz2;    
        derl[i4+1] = dl2;    

        derx[i4+2] = dx3;    
        dery[i4+2] = dy3;    
        derz[i4+2] = dz3;    
        derl[i4+2] = dl3;    

        derx[i4+3] = dx4;    
        dery[i4+3] = dy4;    
        derz[i4+3] = dz4;    
        derl[i4+3] = dl4;    
// USE_DOUBLES
#else
// USE_DOUBLES
        double4 dx = (0.,0.,0.,0.);       
        double4 dy = (0.,0.,0.,0.);       
        double4 dz = (0.,0.,0.,0.);       
        double4 dl = (0.,0.,0.,0.);       

        for (int j = 0; j < stencil_size; j++) {        
            int indx = i*stencil_size + j;
			int ind  = indx << 2;
		// 4 weights ==> 32 bytes (wx,wy,wz,wl) at a point
		// 4 functions ==> 32 bytes
		// 4 derivatives ==> 32 bytes
			int sten = stencils[indx];
			double4 s = (solution[sten], solution[sten+1], solution[sten+2], solution[sten+3]);

			FLOAT w0 = ww[ind];
			FLOAT w1 = ww[ind+1];
			FLOAT w2 = ww[ind+2];
			FLOAT w3 = ww[ind+3];

			dx += s*w0;
			dy += s*w1;
			dz += s*w2;
			dl += s*w3;
		}
		int i4 = i << 2;
		derx[i4] = dx.x; // how can this be efficient? 
		derx[i4+1] = dx.y;
		derx[i4+2] = dx.z;
		derx[i4+3] = dx.w;

		dery[i4] = dy.x;
		dery[i4+1] = dy.y;
		dery[i4+2] = dy.z;
		dery[i4+3] = dy.w;

		derz[i4] = dz.x; 
		derz[i4+1] = dz.y;
		derz[i4+2] = dz.z;
		derz[i4+3] = dz.w;

		derl[i4] = dl.x;
		derl[i4+1] = dl.y;
		derl[i4+2] = dl.z;
		derl[i4+3] = dl.w; // same speed as not using double4s
// USE_DOUBLES
#endif
   }    
}
//----------------------------------------------------------------------
