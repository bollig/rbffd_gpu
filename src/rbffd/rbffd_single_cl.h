#ifndef __RBFFD_SINGLE_CL_H__
#define __RBFFD_SINGLE_CL_H__

//#include <CL/cl.hpp> 
#include "rbffd_cl.h"
#include "utils/opencl/cl_base_class.h"
#include "utils/opencl/structs.h"

class RBFFD_SINGLE_CL : public RBFFD_CL
{
public: 
        RBFFD_SINGLE_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank=0);

	virtual ~RBFFD_SINGLE_CL() { 
            std::cout << "RBFFD_SINGLE_CL Destroyed\n";
	}; 
	
	virtual void applyWeightsForDeriv(DerType which, std::vector<double>& u, std::vector<double>& deriv, bool isChangedU=true)
	{
            std::cout << "[RBFFD_SINGLE_CL] Warning! Using GPU to apply weights, but NOT advance timestep\n";
            unsigned int nb_stencils = grid_ref.getStencilsSize();
            deriv.resize(nb_stencils); 
            // EB: bugfix started index at 0. 
            applyWeightsForDeriv(which, 0, nb_stencils, &u[0], &deriv[0], isChangedU);
	}
		//------------------
        virtual void applyWeightsForDeriv(DerType which, unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU=true) {
            if (useDouble) {
                this->applyWeightsForDerivDouble(which, start_indx, nb_stencils, u, deriv, isChangedU);
            } else {
				printf("SINGLE DISABLED\n");
				exit(0);
            }
        }

		//------------------
        virtual void applyWeightsForDerivDouble(DerType which, unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU=true);


	//void copyResultsToHost(double* deriv);

protected: 
	virtual void allocateGPUMem(); 
};

#endif 
