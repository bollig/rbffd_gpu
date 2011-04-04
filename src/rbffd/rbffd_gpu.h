#ifndef __RBFFD_GPU_H__
#define __RBFFD_GPU_H__

#include <CL/cl.hpp> 
#include "rbffd.h"

class RBFFD_GPU : public RBFFD, public CLBaseClass
{
    protected: 
        // TODO: Pointers to GPU weights
        cl::Buffer[4] gpu_weights; 

        cl::Buffer gpu_stencils; 

        cl::Buffer[4] gpu_deriv_out; 

    public: 
        // Note: dim_num here is the desired dimensions for which we calculate derivatives        
        // (up to 3 right now) 
        //
        //TODO: - constructor should allocate the buffers on the GPU
        //      - onStart applyWeights... will check if(modified) { updateGPUstructs } 
        RBFFD(const Domain& grid, int dim_num, RBF_Type rbf_choice=0); 
        ~RBFFD(); 

        // FIXME: assumes size of buffers does not change (should check if it
        // does and resize accordingly on the GPU. 
        int updateGPUStructs();

        // NOTE: These routines are overridden so we update the GPU
        // appropriately when a new set of weights are calculated (OR call the
        // GPU to calculate weights when we get that done). They can all 3 be
        // optimized in different fashions

        // Compute the full set of derivative weights for all stencils 
        virtual int computeAllWeightsForAllDerivs();
        // Compute the full set of weights for a derivative type
        virtual int computeAllWeightsForDeriv(DerType which); 
        // Compute the full set of derivative weights for a stencil
//TODO:        virtual int computeAllWeightsForStencil(int st_indx); 

        virtual void applyWeightsForDeriv(DerType which, int npts, double* u, double* deriv);


};

#endif 
