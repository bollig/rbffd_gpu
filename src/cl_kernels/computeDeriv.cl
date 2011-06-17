#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

std::string computeDeriv_source = STRINGIFY_WITH_SUBS(

// GPU Only routine
void computeDeriv(       \n
         __global uint* stencils,    \n
         __global FLOAT* weights,   \n
         __global FLOAT* solution,  \n
         __global FLOAT* derivative,    \n
   uint nb_stencils, \n
   uint stencil_size)  \n
{   \n
   uint i = get_global_id(0);    \n
    
   if(i < nb_stencils) {    \n

        FLOAT der = 0.0f;       \n
        for (uint j = 0; j < stencil_size; j++) {        \n
            uint indx = i*stencil_size + j;
//            der += 1. * weights[indx];    \n
            der += solution[stencils[indx]] * weights[indx];    \n
        }   \n
        derivative[i] = der;    \n
   }    \n
}


// GPU Only routine
FLOAT applyWeights1PerThread(       \n
     __global uint* stencil,    \n
     __global FLOAT* st_weights,   \n
     __global FLOAT* solution,  \n
     uint stencil_size)  \n
{   \n
        FLOAT der = 0.0f;       \n
        for (uint j = 0; j < stencil_size; j++) {        \n
            der += solution[stencil[j]] * st_weights[j];    \n
        }   \n
        return der; 
}

);
