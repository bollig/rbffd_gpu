#ifndef __RBFFD_CL_H__
#define __RBFFD_CL_H__

//#include <CL/cl.hpp> 
#include "utils/opencl/cl_base_class.h"
#include "rbffd.h"

class RBFFD_CL : public RBFFD, public CLBaseClass
{
    protected: 
        // Weight buffers matching number of weights we have in super class
        cl::Buffer gpu_weights[NUM_DERIV_TYPES]; 
        double* cpu_weights_d[NUM_DERIV_TYPES]; 
        float* cpu_weights_f[NUM_DERIV_TYPES]; 
        bool deleteCPUWeightsBuffer;

        cl::Buffer gpu_stencils; 
        int*    cpu_stencils;

        cl::Buffer gpu_deriv_out[NUM_DERIV_TYPES]; 


        cl::Buffer gpu_function; 

        // Total size of the gpu-stencils buffer. This should also be the size
        // of a single element of gpu_weights array. 
        size_t gpu_stencil_size; 

        // number of bytes for: 
        //      - gpu_stencils
        //      - gpu_deriv_out[ i ]
        //      - gpu_weights[ i ]
        //      - gpu_function
        size_t stencil_mem_bytes;
        size_t deriv_mem_bytes;
        size_t weights_mem_bytes;
        size_t function_mem_bytes;

        // Is a double precision extension available on the unit? 
        bool useDouble; 



    public: 
        // Note: dim_num here is the desired dimensions for which we calculate derivatives        
        // (up to 3 right now) 
        //
        //TODO: - constructor should allocate the buffers on the GPU
        //      - onStart applyWeights... will check if(modified) { updateGPUstructs } 

        RBFFD_CL(Grid* grid, int dim_num, int rank=0);

        virtual ~RBFFD_CL() { if (deleteCPUWeightsBuffer) { this->clearCPUWeights();} }; 

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
        virtual void applyWeightsForDeriv(DerType which, std::vector<double>& u, std::vector<double>& deriv, bool isChangedU=true) { 
//            std::cout << "GPU: ";
            size_t nb_stencils = grid_ref.getStencilsSize();
            deriv.resize(nb_stencils); 
            applyWeightsForDeriv(which, grid_ref.getNodeListSize(), nb_stencils, &u[0], &deriv[0], isChangedU);
        }
        virtual void applyWeightsForDeriv(DerType which, size_t nb_nodes, size_t nb_stencils, double* u, double* deriv, bool isChangedU=true) {
            if (useDouble) {
                this->applyWeightsForDerivDouble(which, nb_nodes, nb_stencils, u, deriv, isChangedU);
            } else {
                this->applyWeightsForDerivSingle(which, nb_nodes, nb_stencils, u, deriv, isChangedU);
            }
        }

        virtual void applyWeightsForDerivDouble(DerType which, size_t nb_nodes, size_t nb_stencils, double* u, double* deriv, bool isChangedU=true);

        virtual void applyWeightsForDerivSingle(DerType which, size_t nb_nodes, size_t nb_stencils, double* u, double* deriv, bool isChangedU=true);

    protected: 
        void setupTimers(); 
        void loadKernel(); 
        void allocateGPUMem(); 

        void clearCPUWeights();

        // forceFinish ==> should we fire a queue.finish() and make sure all
        // tasks are completed (synchronously) before returning
        void updateStencils(bool forceFinish);
        
        void updateWeights(bool forceFinish)
        { 
            if (useDouble) { updateWeightsDouble(forceFinish); 
            } else { updateWeightsSingle(forceFinish); }
        }
        void updateFunction(size_t nb_nodes, double* u, bool forceFinish)
        { 
            if (useDouble) { updateFunctionDouble(nb_nodes, u, forceFinish); 
            } else { updateFunctionSingle(nb_nodes, u, forceFinish); }
        }

        void updateWeightsDouble(bool forceFinish);
        void updateWeightsSingle(bool forceFinish);
        void updateFunctionDouble(size_t nb_nodes, double* u, bool forceFinish);
        void updateFunctionSingle(size_t nb_nodes, double* u, bool forceFinish);
};

#endif 
