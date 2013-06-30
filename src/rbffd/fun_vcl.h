#ifndef __FUN_VCL_H__
#define __FUN_VCL_H__

// Test new super arrays

#include "rbffd_vcl.h"
#include "utils/opencl/structs.h"

class FUN_VCL : public RBFFD_VCL
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
		  FUN4_DERIV4_WEIGHT4,  // 4 functions, 4 derivatives, using double4 on GPU, stencil node = fastest index
		  FUN4_DERIV4_WEIGHT4_INV}; // 4 fun, 4 deriv, using double4 on GPU, rbf node = fastest index.
		KernelType kernel_type; // poor name

	public:
        FUN_VCL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank=0);

		// Could use Boost shared poitners
        virtual ~FUN_VCL() { 
            //if (deleteCPUWeightsBuffer) { this->clearCPUWeights();} 
            //if (deleteCPUNodesBuffer) { this->clearCPUNodes();}
            //if (deleteCPUStencilsBuffer) { this->clearCPUStencils();}
            std::cout << "FUN_VCL Destroyed\n";
        } 

		void setKernelType(KernelType kernel_type_);

		//------------------
		// Should be changed so I call GPU and CPU functions with rbffd, and run on GPU and CPU
#if 0
		virtual void computeDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, 
			SuperBuffer<double>& deriv_y, SuperBuffer<double>& deriv_z, SuperBuffer<double>& deriv_l, bool isChangedU);

		virtual void computeDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, bool isChangedU);
		virtual void convertWeights();
		//------------------
#endif
    protected: 
        virtual void allocateGPUMem(); 
};

#endif 