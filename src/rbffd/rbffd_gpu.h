#ifndef __RBFFD_GPU_H__
#define __RBFFD_GPU_H__

//#include <CL/cl.hpp> 
#include "utils/opencl/cl_base_class.h"
#include "rbffd.h"

class RBFFD_GPU : public RBFFD, public CLBaseClass
{
    protected: 
        // TODO: Pointers to GPU weights
        cl::Buffer gpu_weights[4]; 

        cl::Buffer gpu_stencils; 
        size_t*    cpu_stencils;

        cl::Buffer gpu_deriv_out[4]; 

        // Total size of the gpu-stencils buffer. This should also be the size
        // of a single element of gpu_weights array. 
        size_t gpu_stencil_size; 
        // Total number of bytes allocated for stencil (i.e., gpu_stencil_size*sizeof(float|double))
        size_t stencil_mem_size; 
    
        // Is a double precision extension available on the unit? 
        bool useDouble; 

    public: 
        // Note: dim_num here is the desired dimensions for which we calculate derivatives        
        // (up to 3 right now) 
        //
        //TODO: - constructor should allocate the buffers on the GPU
        //      - onStart applyWeights... will check if(modified) { updateGPUstructs } 

        RBFFD_GPU(Domain* grid, int dim_num, int rank=0);

        virtual ~RBFFD_GPU() { /*noop*/ }; 

        // FIXME: assumes size of buffers does not change (should check if it
        // does and resize accordingly on the GPU. 
        //TODO:        int updateGPUStructs();

        // NOTE: These routines are overridden so we update the GPU
        // appropriately when a new set of weights are calculated (OR call the
        // GPU to calculate weights when we get that done). They can all 3 be
        // optimized in different fashions

        // Compute the full set of derivative weights for all stencils 
        //TODO:        virtual int computeAllWeightsForAllDerivs();
        // Compute the full set of weights for a derivative type
        //TODO:        virtual int computeAllWeightsForDeriv(DerType which); 
        // Compute the full set of derivative weights for a stencil
        //TODO:        virtual int computeAllWeightsForStencil(int st_indx); 

        // FIXME: HACK--> this routine is called in a situation where we want to access a superclass routine inside. 
        //                This override is how we hack this together.
        // Apply weights to an input solution vector and get the corresponding derivatives out
        virtual void applyWeightsForDeriv(DerType which, std::vector<double>& u, std::vector<double>& deriv) { 
            std::cout << "GPU: ";
            deriv.resize(u.size()); 
            applyWeightsForDeriv(which, u.size(), &u[0], &deriv[0]);
        }
        virtual void applyWeightsForDeriv(DerType which, int npts, double* u, double* deriv);


    protected: 
        void setupTimers(); 
        void loadKernel(); 
        void allocateGPUMem(); 
        void updateStencils();

};

#endif 
