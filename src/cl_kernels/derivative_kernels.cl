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
   const size_t nb_stencils, \n
   const size_t stencil_size)  \n
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
       // for (size_t j = 0; j < 1/*stencil_size*/; j++) {        \n
            size_t j = 0; 
            size_t indx = i*stencil_size + j;// + (i * stencil_size); 
 //               der += stencils[stencil_size * i + j] * weight[j];    \n

                //der += stencils[3];//stencils[j+i*stencil_size]; 
                der = stencils[indx]; 
                //der += solution[stencil[j]] * weight[j];    \n
       // }   \n
        derivative[i] = der;    \n
   }    \n
}

);
