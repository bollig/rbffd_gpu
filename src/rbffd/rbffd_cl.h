#ifndef __RBFFD_CL_H__
#define __RBFFD_CL_H__

#include "rbffd.h"
//#include <CL/cl.hpp> 
#include "utils/opencl/cl_base_class.h"
#include "utils/opencl/structs.h"

class RBFFD_CL : public RBFFD, public CLBaseClass
{
    protected: 
        // Weight buffers matching number of weights we have in super class
        cl::Buffer gpu_weights[NUM_DERIVATIVE_TYPES]; 
        cl::Buffer gpu_nodes;

        double* cpu_weights_d[NUM_DERIVATIVE_TYPES];
        float* cpu_weights_f[NUM_DERIVATIVE_TYPES];
        double4* cpu_nodes;
        bool deleteCPUWeightsBuffer;
        bool deleteCPUNodesBuffer;
        bool deleteCPUStencilsBuffer;

        cl::Buffer gpu_stencils; 
        unsigned int*    cpu_stencils;

        cl::Buffer gpu_deriv_out; 

        cl::Buffer gpu_function; 

        // Total size of the gpu-stencils buffer. This should also be the size
        // of a single element of gpu_weights array. 
        unsigned int gpu_stencil_size; 

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

        // This will be either the MAX_STENCIL_SIZE (computed by
        // GridInterface), or the stencil size rounded to nearest
        // multiple of 16 or 32. Any stencils that do not meet the
        // stencil_padded_size are padded with 0s for weights and 
        // the stencil center index for the padded indices. 
        unsigned int stencil_padded_size; 

    public: 
        // Note: dim_num here is the desired dimensions for which we calculate derivatives        
        // (up to 3 right now) 
        //
        //TODO: - constructor should allocate the buffers on the GPU
        //      - onStart applyWeights... will check if(modified) { updateGPUstructs } 

        RBFFD_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank=0);

        virtual ~RBFFD_CL() { 
            if (deleteCPUWeightsBuffer) { this->clearCPUWeights();} 
            if (deleteCPUNodesBuffer) { this->clearCPUNodes();}
            if (deleteCPUStencilsBuffer) { this->clearCPUStencils();}
            std::cout << "RBFFD_CL Destroyed\n";
        }; 


        cl::Buffer& getGPUStencils() { return gpu_stencils; }
        cl::Buffer& getGPUNodes() { return gpu_nodes; }
        cl::Buffer& getGPUWeights(DerType which) { return gpu_weights[getDerTypeIndx(which)]; }


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
            std::cout << "[RBFFD_CL] Warning! Using GPU to apply weights, but NOT advance timestep\n";
            unsigned int nb_stencils = grid_ref.getStencilsSize();
            deriv.resize(nb_stencils); 
            //applyWeightsForDeriv(which, grid_ref.getNodeListSize(), nb_stencils, &u[0], &deriv[0], isChangedU);
            // EB: bugfix started index at 0. 
            applyWeightsForDeriv(which, 0, nb_stencils, &u[0], &deriv[0], isChangedU);
        }
		//------------------
        virtual void applyWeightsForDeriv(DerType which, unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU=true) {
            if (useDouble) {
                this->applyWeightsForDerivDouble(which, start_indx, nb_stencils, u, deriv, isChangedU);
            } else {
                this->applyWeightsForDerivSingle(which, start_indx, nb_stencils, u, deriv, isChangedU);
            }
        }

		//------------------
        virtual void applyWeightsForDerivDouble(DerType which, unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU=true);

        virtual void applyWeightsForDerivSingle(DerType which, unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU=true);

        // forceFinish ==> should we fire a queue.finish() and make sure all
        // tasks are completed (synchronously) before returning
        void updateStencilsOnGPU(bool forceFinish);
        
        void updateWeightsOnGPU(bool forceFinish)
        { 
            if (useDouble) { updateWeightsDouble(forceFinish); 
            } else { updateWeightsSingle(forceFinish); }
        }
		//-----------------
        void updateFunctionOnGPU(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish)
        { 
            if (useDouble) { updateFunctionDouble(start_indx, nb_vals, u, forceFinish); 
            } else { updateFunctionSingle(start_indx, nb_vals, u, forceFinish); }
        }

        void updateNodesOnGPU(bool forceFinish);

		//-----------------
        bool areGPUKernelsDouble() { return useDouble; }
        
        unsigned int getStencilPaddedSize() {
            return stencil_padded_size;
        }

	void setAlignWeights(bool alignYN) { alignWeights = alignYN; } 

    protected: 
        void setupTimers(); 
        void loadKernel(); 
        void allocateGPUMem(); 

        void clearCPUWeights();
        void clearCPUStencils();
        void clearCPUNodes();

        void updateWeightsDouble(bool forceFinish);
        void updateWeightsSingle(bool forceFinish);
        void updateFunctionDouble(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish);
        void updateFunctionSingle(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish);


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
