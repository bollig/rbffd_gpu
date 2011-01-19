#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#define EXAMPLE_MACRO 4

std::string kernel_source = STRINGIFY_WITH_SUBS(

// Assuming that our stencils are uniform in size for now
__kernel void           \n
computeDerivKernel(             \n
         __global int* stencils,    \n
         __global float* weights,   \n
         __global float* solution,  \n
         __global float* derivative,    \n
   int nb_stencils, \n
   int stencil_size)  \n
{   \n
   size_t i = get_global_id(0);    \n
    
   if(i < nb_stencils) {    \n
  //      int offset = stencil_offset[i];     \n
   //     int offset_stop = stencil_offset[i+1];  \n 
   //     int stencil_size = offset_stop - offset;    \n

        // our stencil starts at the offset \n
        //__global int* stencil = stencils + stencil_size * i;    \n
       // __global float* weight = weights + stencil_size * i;    \n


        float der = 0.0f;       \n
        for (int j = 0; j < stencil_size; j++) {        \n
      //      int j = stencil_size - 3; 
            size_t indx = i*stencil_size + j;
 //               der += stencils[stencil_size * i + j] * weight[j];    \n
                //der += stencils[3];//stencils[j+i*stencil_size]; 
//                der = solution[stencils[indx]]; 
  //             der = solution[stencils[indx]] * weights[indx];    \n
            der += weights[indx];
        }   \n
        derivative[i] = der;    \n
   }    \n
}

);
