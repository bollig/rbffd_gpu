//#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
//#define STRINGIFY(s) #s

#include "useDouble.cl"

//std::string computeDeriv_source = STRINGIFY_WITH_SUBS(

// GPU Only routine
void computeDeriv(       
         __global int* stencils,    
         __global FLOAT* weights,   
         __global FLOAT* solution,  
         __global FLOAT* derivative,    
   int nb_stencils, 
   int stencil_size)  
{   
   int i = get_global_id(0);    
   if(i < nb_stencils) {    
        FLOAT der = 0.0f;       
        for (int j = 0; j < stencil_size; j++) {       
            int indx = i*stencil_size + j;
//            der += 1. * weights[indx];    
        //    der += 1. ;    
            der += solution[stencils[indx]] * weights[indx];    
        }   
        derivative[i] = der;    
   }    
}


// GPU Only routine
FLOAT applyWeights1PerThread(       
     __global int* stencil,    
     __global FLOAT* st_weights,   
     __global FLOAT* solution,  
     int stencil_size)  
{   
        FLOAT der = 0.0f;       
        for (int j = 0; j < stencil_size; j++) { 
            der += solution[stencil[j]] * st_weights[j];    
        }   
        return der; 
}

//);
