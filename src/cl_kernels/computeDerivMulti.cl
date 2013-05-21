#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

std::string computeDerivMulti_source = STRINGIFY_WITH_SUBS(

// GPU Only routine
void computeDerivMulti(       \n
         __global uint* stencils,    \n // double4
         __global FLOAT* wx,   \n // double4
         __global FLOAT* wy,   \n // double4
         __global FLOAT* wz,   \n // double4
         __global FLOAT* wl,   \n // double4
         __global FLOAT* solution,  \n // (has n FLOATS)
         __global FLOAT* derx,    \n // double4
         __global FLOAT* dery,    \n // double4
         __global FLOAT* derz,    \n // double4
         __global FLOAT* derl,    \n // double4
   uint nb_stencils, \n
   uint stencil_size)  \n
{   \n
   uint i = get_global_id(0);    \n

   //double4 xxx = (0.,0.,0.,0.); // VALID EXPRESSION
   // put solution into double4; have single thread work with double4
   // Best to have one thread per stencil value and have them access individual elements of stencil
    
   if(i < nb_stencils) {    \n

        FLOAT dx = 0.0;       \n
        FLOAT dy = 0.0;       \n
        FLOAT dz = 0.0;       \n
        FLOAT dl = 0.0;       \n
		// point "i" handled by a single thread
		// solution is "reused", but is there a guarantee it will remain in cache? 
		// would it be possible to assign a different thread to each derivative? 
        for (uint j = 0; j < stencil_size; j++) {        \n
            uint indx = i*stencil_size + j;
        //    der += 1. * weights[indx];    \n
        //    der += 1. ;    \n
			FLOAT sol = solution[stencils[indx]];
            dx += sol * wx[indx];    \n
            dy += sol * wy[indx];    \n
            dz += sol * wz[indx];    \n
            dl += sol * wl[indx];    \n
        }   \n
        derx[i] = dx;    \n
        dery[i] = dy;    \n
        derz[i] = dz;    \n
        derl[i] = dl;    \n
   }    \n
}
//----------------------------------------------------------------------
// compute x,y,z,l derivatives of (u,v,w,p)
// sol: (u,v,w,p)[1], (u,v,w,p)[2], ...
// derx: (dudx,dvdx,dwdx,dpdx)[1], ...
// dery: (dudy,dvdy,dwdy,dpdy)[1], ...
void computeDerivMulti4(       \n
         __global uint* stencils,    \n // double4
         __global FLOAT* wx,   \n // double4
         __global FLOAT* wy,   \n // double4
         __global FLOAT* wz,   \n // double4
         __global FLOAT* wl,   \n // double4
         __global FLOAT* solution,  \n // (has 4n FLOATS)
         __global FLOAT* derx,    \n // double4
         __global FLOAT* dery,    \n // double4
         __global FLOAT* derz,    \n // double4
         __global FLOAT* derl,    \n // double4
   uint nb_stencils, \n
   uint stencil_size)  \n
{   \n
   uint i = get_global_id(0);    \n


   // One thread takes care of one stencil
    
   if(i < nb_stencils) {    \n

        FLOAT dx = 0.0;       \n
        FLOAT dy = 0.0;       \n
        FLOAT dz = 0.0;       \n
        FLOAT dl = 0.0;       \n
		// point "i" handled by a single thread
		// solution is "reused", but is there a guarantee it will remain in cache? 
		// would it be possible to assign a different thread to each derivative? 
        for (uint j = 0; j < stencil_size; j++) {        \n
            uint indx = i*stencil_size + j;
        //    der += 1. * weights[indx];    \n
        //    der += 1. ;    \n
			FLOAT sol = solution[stencils[indx]];
            dx += sol * wx[indx];    \n
            dy += sol * wy[indx];    \n
            dz += sol * wz[indx];    \n
            dl += sol * wl[indx];    \n
        }   \n
        derx[i] = dx;    \n
        dery[i] = dy;    \n
        derz[i] = dz;    \n
        derl[i] = dl;    \n
   }    \n
}
//----------------------------------------------------------------------

//------------------------------
//------------------------------

);
