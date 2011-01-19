#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#define EXAMPLE_MACRO 4

std::string kernel_source = STRINGIFY_WITH_SUBS(

// Assuming that our stencils are uniform in size for now
__kernel void           \n
computeDerivKernel(       \n
         __global int* stencils,    \n
         __global float* weights,   \n
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

);
