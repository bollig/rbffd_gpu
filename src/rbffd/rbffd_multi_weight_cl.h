#ifndef __RBFFD_MULTI_WEIGHT_CL_H__
#define __RBFFD_MULTI_WEIGHT_CL_H__

//#include <CL/cl.hpp> 
#include "utils/opencl/cl_base_class.h"
//#include "rbffd.h"
#include "rbffd_cl.h"
#include "utils/opencl/structs.h"

class RBFFD_MULTI_WEIGHT_CL : public RBFFD_CL
{
    public: 
        // Note: dim_num here is the desired dimensions for which we calculate derivatives        
        // (up to 3 right now) 
        //
        //TODO: - constructor should allocate the buffers on the GPU
        //      - onStart applyWeights... will check if(modified) { updateGPUstructs } 

        RBFFD_MULTI_WEIGHT_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank=0);

        virtual ~RBFFD_MULTI_WEIGHT_CL() { 
            if (deleteCPUWeightsBuffer) { this->clearCPUWeights();} 
            if (deleteCPUNodesBuffer) { this->clearCPUNodes();}
            if (deleteCPUStencilsBuffer) { this->clearCPUStencils();}
            std::cout << "RBFFD_MULTI_WEIGHT_CL Destroyed\n";
        }; 


        // Compute the full set of derivative weights for a stencil
        //TODO:        virtual int computeAllWeightsForStencil(int st_indx); 

        // FIXME: HACK--> this routine is called in a situation where we want to access a superclass routine inside. 
        //                This override is how we hack this together.
        // Apply weights to an input solution vector and get the corresponding derivatives out
		// Does not exist in base class.
        virtual void applyWeightsForDeriv(std::vector<double>& u, 
				std::vector<double>& deriv_x, 
				std::vector<double>& deriv_y, 
				std::vector<double>& deriv_z, 
				std::vector<double>& deriv_l, 
				bool isChangedU=true)
		{
            std::cout << "[RBFFD_MULTI_WEIGHT_CL] Warning! Using GPU to apply weights, but NOT advance timestep\n";
            unsigned int nb_stencils = grid_ref.getStencilsSize();
            deriv_x.resize(4*nb_stencils); 
            deriv_y.resize(4*nb_stencils); 
            deriv_z.resize(4*nb_stencils); 
            deriv_l.resize(4*nb_stencils); 
            //applyWeightsForDeriv(which, grid_ref.getNodeListSize(), nb_stencils, &u[0], &deriv[0], isChangedU);
            // EB: bugfix started index at 0. 
            applyWeightsForDeriv(0, nb_stencils, &u[0], &deriv_x[0], &deriv_y[0], &deriv_z[0], &deriv_l[0], isChangedU);
        }
		//------------------
        virtual void applyWeightsForDeriv(unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv_x, double* deriv_y, double* deriv_z, double* deriv_l, bool isChangedU=true) {
            if (useDouble) {
                this->applyWeightsForDerivDouble(start_indx, nb_stencils, u, deriv_x, deriv_y, deriv_z, deriv_l, isChangedU);
            } else {
				printf("SINGLE DISABLED\n");
				exit(0);
				;
            }
        }

        virtual void applyWeightsForDerivDouble(unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv_x, double* deriv_y, double* deriv_z, double* deriv_l, bool isChangedU=true);


        //virtual void updateFunctionDouble(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish);


    protected: 
        virtual void allocateGPUMem(); 
		virtual void updateWeightsDouble(bool forceFinish);

};

#endif 
