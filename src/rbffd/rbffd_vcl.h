#ifndef __RBFFD_VCL_H__
#define __RBFFD_VCL_H__

//#include <CL/cl.hpp> 
//#include "utils/opencl/cl_base_class.h"
#include "rbffd.h"
#include "utils/opencl/structs.h"

#include "utils/opencl/viennacl_typedefs.h"

class RBFFD_VCL : public RBFFD
{    protected: 
        // Weight buffers matching number of weights we have in super class
        VCL_ELL_MAT_t* gpu_weights[NUM_DERIVATIVE_TYPES]; 
        // Nodes are X, Y, Z vectors (will transform to theta, lambda)
        VCL_ELL_MAT_t* gpu_all_weights;
        VCL_VEC4_t* gpu_nodes;

        UBLAS_MAT_t* cpu_weights_d[NUM_DERIVATIVE_TYPES];
        UBLAS_VEC4_t* cpu_nodes;
        bool deleteCPUWeightsBuffer;
        bool deleteCPUNodesBuffer;

        unsigned int*    cpu_stencils;

        VCL_VEC_t* gpu_deriv_out; 

        VCL_VEC_t* gpu_function; 

        unsigned int gpu_nnz; 

        // number of bytes for: 
        //      - gpu_stencils
        //      - gpu_deriv_out[ i ]
        //      - gpu_weights[ i ]
        //      - gpu_function
        unsigned int stencil_mem_bytes;
        unsigned int deriv_mem_bytes;
        unsigned int weights_mem_bytes;
        unsigned int function_mem_bytes;
        unsigned int nodes_mem_bytes;

        // Is a double precision extension available on the unit? 
        bool useDouble; 

        // Set this to control the weight padding/alignment 
        bool alignWeights; 
        unsigned int alignMultiple;

    public: 
        // Note: dim_num here is the desired dimensions for which we calculate derivatives        
        // (up to 3 right now) 
        //
        //TODO: - constructor should allocate the buffers on the GPU
        //      - onStart applyWeights... will check if(modified) { updateGPUstructs } 

        RBFFD_VCL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank=0);

        virtual ~RBFFD_VCL() { 
            if (deleteCPUWeightsBuffer) { this->clearCPUWeights();} 
            if (deleteCPUNodesBuffer) { this->clearCPUNodes();}
            this->tm.printAll(); 
            this->tm.clear();
            std::cout << "RBFFD_VCL destroyed\n";
        }; 


        // This should be: 
        VCL_ELL_MAT_t* getGPUWeights(DerType which) { return gpu_weights[getDerTypeIndx(which)]; }
        VCL_ELL_MAT_t* getGPUWeights(DerTypeIndx which_i) { return gpu_weights[which_i]; }


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
            std::cout << "[RBFFD_VCL] Warning! Using GPU to apply weights, but NOT advance timestep\n";
            unsigned int nb_stencils = grid_ref.getStencilsSize();
            deriv.resize(nb_stencils); 
            applyWeightsForDeriv(which, grid_ref.getNodeListSize(), nb_stencils, &u[0], &deriv[0], isChangedU);
        }
        virtual void applyWeightsForDeriv(DerType which, unsigned int nb_nodes, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU=true) {
                this->applyWeightsForDerivDouble(which, nb_nodes, nb_stencils, u, deriv, isChangedU);
        }

        virtual void applyWeightsForDerivDouble(DerType which, unsigned int nb_nodes, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU=true);

        virtual void applyWeightsForDeriv(DerType which, VCL_VEC_t& u, VCL_VEC_t& deriv, bool isChangedU=true); 

        // forceFinish ==> should we fire a queue.finish() and make sure all
        // tasks are completed (synchronously) before returning
        void updateWeightsOnGPU(bool forceFinish)
        { 
                updateWeightsDouble(forceFinish); 
        }
        void updateFunctionOnGPU(unsigned int nb_nodes, double* u, bool forceFinish)
        { 
                updateFunctionDouble(nb_nodes, u, forceFinish); 
        }

        void updateNodesOnGPU(bool forceFinish);

        bool areGPUKernelsDouble() { return useDouble; }
        

    protected: 
        void setupTimers(); 
        void loadKernel(); 
        void allocateGPUMem(); 

        void clearCPUWeights();
        void clearCPUNodes();

        void updateWeightsDouble(bool forceFinish);
        void updateFunctionDouble(unsigned int nb_nodes, double* u, bool forceFinish);


    protected: 
        unsigned int getNextMultiple(unsigned int stencil_size) {
            unsigned int nearest = alignMultiple; 
            while (stencil_size > nearest) 
                nearest += alignMultiple; 
            return nearest;
        }
        
        unsigned int getNextMultipleOf32(unsigned int stencil_size) {
            unsigned int nearest = 32; 
            while (stencil_size > nearest) 
                nearest += 32; 
            return nearest;
        }
};

#endif 
