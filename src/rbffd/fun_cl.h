#ifndef __FUN_CL_H__
#define __FUN_CL_H__

// Test new super arrays

#include "rbffd_cl.h"
#include "utils/opencl/structs.h"

class FUN_CL : public RBFFD_CL
{
	private:
		int nb_nodes;
		int nodes_per_stencil;
    	int nb_stencils;

    public: 
	// FUN1_DERIV4_WEIGHT4: 1 function, 4 derivatives, use double4
	// FUN1_DERIV1_WEIGHT4: 1 function, 4 derivatives, use double
		enum KernelType {FUN_KERNEL, FUN_INV_KERNEL, FUN_DERIV4_KERNEL, 
		  FUN1_DERIV4_WEIGHT4, FUN1_DERIV1_WEIGHT4,
		  FUN4_DERIV1_WEIGHT4};  // 4 functions, 4 derivatives, using double on GPU
		KernelType kernel_type; // poor name

	public:
        FUN_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank=0);

		// Could use Boost shared poitners
        virtual ~FUN_CL() { 
            if (deleteCPUWeightsBuffer) { this->clearCPUWeights();} 
            if (deleteCPUNodesBuffer) { this->clearCPUNodes();}
            if (deleteCPUStencilsBuffer) { this->clearCPUStencils();}
            std::cout << "FUN_CL Destroyed\n";
        } 

		void setKernelType(KernelType kernel_type_);

		//------------------
		// Should be changed so I call GPU and CPU functions with rbffd, and run on GPU and CPU
		virtual void computeDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, 
			SuperBuffer<double>& deriv_y, SuperBuffer<double>& deriv_z, SuperBuffer<double>& deriv_l, bool isChangedU);

		virtual void computeDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, bool isChangedU);
		virtual void convertWeights();
		//------------------

    protected: 
        virtual void allocateGPUMem(); 
};

#endif 
