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
    	unsigned int nb_stencils;

    public: 
        FUN_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank=0);

		// Could use Boost shared poitners
        virtual ~FUN_CL() { 
            if (deleteCPUWeightsBuffer) { this->clearCPUWeights();} 
            if (deleteCPUNodesBuffer) { this->clearCPUNodes();}
            if (deleteCPUStencilsBuffer) { this->clearCPUStencils();}
            std::cout << "FUN_CL Destroyed\n";
        } 

		//------------------
		// Should be changed so I call GPU and CPU functions with rbffd, and run on GPU and CPU
		virtual void calcDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, 
			SuperBuffer<double>& deriv_y, SuperBuffer<double>& deriv_z, SuperBuffer<double>& deriv_l, bool isChangedU);
		//------------------

    protected: 
        virtual void allocateGPUMem(); 
};

#endif 
