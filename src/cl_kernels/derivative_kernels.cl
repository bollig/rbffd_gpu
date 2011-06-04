#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s


std::string kernel_source = STRINGIFY_WITH_SUBS(


// NOTE: my CLBaseClass can add FP64 support. It looks at the
// available device extensions to know which one it should enable
// at runtime (this works on both ATI and NVidia)

// Assuming that our stencils are uniform in size for now
__kernel void           \n
computeDerivKernelFLOAT(       \n
         __global int* stencils,    \n
         __global double* weights,   \n
         __global float* solution,  \n
         __global float* derivative,    \n
   int nb_stencils, \n
   int stencil_size)  \n
{   \n
   size_t i = get_global_id(0);    \n
    
   if(i < nb_stencils) {    \n

        float der = 0.0f;       \n
        for (int j = 0; j < stencil_size; j++) {        \n
            size_t indx = i*stencil_size + j;
//            der += 1. * weights[indx];    \n
            der += solution[stencils[indx]] * weights[indx];    \n
        }   \n
        derivative[i] = der;    \n
   }    \n
}




// Assuming that our stencils are uniform in size for now
__kernel void computeDerivKernelDOUBLE(       \n
         __global int* stencils,    \n
         __global double* weights,   \n
         __global double* solution,  \n
         __global double* derivative,    \n
   int nb_stencils, \n
   int stencil_size)  \n
{   \n
   size_t i = get_global_id(0);    \n
    
   if(i < nb_stencils) {    \n

        double der = 0.0f;       \n
        for (int j = 0; j < stencil_size; j++) {        \n
            size_t indx = i*stencil_size + j;
//            der += 1. * weights[indx];    \n
            der += solution[stencils[indx]] * weights[indx];    \n
        }   \n
        derivative[i] = der;    \n
   }    \n
}

);
