#ifndef __RBFFD_CL_H__
#define __RBFFD_CL_H__

//#include <CL/cl.hpp> 
#include <string>
#include <vector>
#include "utils/opencl/cl_base_class.h"
#include "rbffd.h"
#include "utils/opencl/structs.h"


class RBFFD_CL : public RBFFD, public CLBaseClass
{
    protected: 
        // Weight buffers matching number of weights we have in super class
        cl::Buffer gpu_weights[NUM_DERIVATIVE_TYPES]; 
        cl::Buffer gpu_all_weights; // concatenate the weights
        cl::Buffer gpu_nodes;

        double* cpu_weights_d[NUM_DERIVATIVE_TYPES];
        float* cpu_weights_f[NUM_DERIVATIVE_TYPES];
        double4* cpu_nodes;
        bool deleteCPUWeightsBuffer;
        bool deleteCPUNodesBuffer;
        bool deleteCPUStencilsBuffer;

        cl::Buffer gpu_stencils; 
        unsigned int*    cpu_stencils;

		// for use by various subclasses. 
        cl::Buffer gpu_deriv_out; 
        cl::Buffer gpu_deriv_x_out; 
        cl::Buffer gpu_deriv_y_out; 
        cl::Buffer gpu_deriv_z_out; 
        cl::Buffer gpu_deriv_l_out; 
        cl::Buffer gpu_function; 

		#if 0
		SuperBuffer<double> sup_deriv;
		SuperBuffer<double> sup_deriv_x;
		SuperBuffer<double> sup_deriv_y;
		SuperBuffer<double> sup_deriv_z;
		SuperBuffer<double> sup_deriv_l;
		SuperBuffer<double> sup_function;
		SuperBuffer<double> sup_weights[NUM_DERIVATIVE_TYPES];
		#endif
		SuperBuffer<double> sup_all_weights;
		SuperBuffer<int>    sup_stencils;

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
    	unsigned int all_weights_bytes;
    	unsigned int bytesAllocated;

        // Is a double precision extension available on the unit? 
        bool useDouble; 


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
			printf("Enter RBFFD_CL destructor\n");
            if (deleteCPUWeightsBuffer) { this->clearCPUWeights();} 
            if (deleteCPUNodesBuffer) { this->clearCPUNodes();}
            if (deleteCPUStencilsBuffer) { this->clearCPUStencils();}
			printf("Exit RBFFD_CL destructor\n");
        } 


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
			printf("[RBFFD_CL] INSIDE applyWeightsForDeriv\n");
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
        virtual void updateStencilsOnGPU(bool forceFinish);
        
        virtual void updateWeightsOnGPU(bool forceFinish)
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

        virtual void updateNodesOnGPU(bool forceFinish);

		//-----------------
        bool areGPUKernelsDouble() { return useDouble; }
        
        unsigned int getStencilPaddedSize() {
            return stencil_padded_size;
        }

	virtual void setAlignWeights(bool alignYN) { alignWeights = alignYN; } 

public:

	virtual void computeDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, 
			SuperBuffer<double>& deriv_y, SuperBuffer<double>& deriv_z, SuperBuffer<double>& deriv_l, bool isChangedU)
		    { 
				printf("[RBFFD] enter computeDeriv, 5 SuperBuffer args\n");
			}

	virtual void computeDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, bool isChangedU)
		    { 
				printf("[RBFFD] enter computeDeriv, 2 SuperBuffer args\n");
			}

    protected: 
        void setupTimers(); 
        //virtual void loadKernel(); 
        virtual void loadKernel(const std::string& kernel, const std::string& filename); 
        virtual void allocateGPUMem(); 

        virtual void clearCPUWeights();
        virtual void clearCPUStencils();
        virtual void clearCPUNodes();

        virtual void updateWeightsDouble(bool forceFinish);
        virtual void updateWeightsSingle(bool forceFinish);
        virtual void updateFunctionDouble(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish);
        virtual void updateFunctionSingle(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish);

		virtual void enqueueKernel(const cl::Kernel& kernel, const cl::NDRange& tot_work_items, 
				const cl::NDRange& items_per_workgroup, bool is_finish);


    protected: 
		int getSize(cl::Buffer& buf) {
			int mem_size;
			try {
				mem_size = buf.getInfo<CL_MEM_SIZE>();
	 		} catch (cl::Error er) {
	    		printf("[cl::Buffer] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
	 			mem_size = -1; // invalid object
			}
			return(mem_size);
		}
		// copy from GPU to CPU (device to host)
		// only copy if gpu buffer has nonzero size
		// should check size of host array to ensure it that there is  sufficienty space
		template <typename T>
		void copyArrayToHost(cl::Buffer& buf, T* host_address) {
			int mem_size = getSize(buf);
			if (mem_size < 1) return;
			// do not use monitoring events
    		err = queue.enqueueReadBuffer(buf, CL_TRUE, 0, mem_size, host_address, NULL, NULL);
			if (err != CL_SUCCESS) {
				std::cerr << " enqueueRead ERROR: " << err << std::endl;
			}
		}
		
		// copy from CPU to GPU (host to device)
		// only copy if gpu buffer has nonzero size
		// should check size of host array to ensure it that there is  sufficienty space
		// Assumes GPU and cpu buffers have the same size
		// It would be best to have a wrapper around buffers to be used both on GPU and CPU
		// If type of GPU array is different from type of array on CPU, e.g., double on CPU and float on GPU, 
		//    copy to a temporary array first (NOT DONE)
		template <typename T>
		void copyArrayToGPU(T* host_address, cl::Buffer& buf) {
			int mem_size = getSize(buf);
			printf("mem_size= %d\n", mem_size);
			if (mem_size < 1) return;
			// do not use monitoring events
    		err = queue.enqueueWriteBuffer(buf, CL_TRUE, 0, mem_size, host_address, NULL, NULL);
			if (err != CL_SUCCESS) {
				std::cerr << " enqueueRead ERROR: " << err << std::endl;
			}
		}


};

#endif 
