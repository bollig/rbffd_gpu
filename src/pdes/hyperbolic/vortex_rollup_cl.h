#ifndef __VORTEX_ROLLUP_CL_H__
#define __VORTEX_ROLLUP_CL_H__

#include "pdes/time_dependent_pde_cl.h"


// TODO: extend this class and compute diffusion in two terms: lapl(y(t)) = div(y(t)) .dot. grad(y(t))
class VortexRollup_CL : public TimeDependentPDE_CL
{
    public: 
        VortexRollup_CL(Domain* grid, RBFFD_CL* der, Communicator* comm, int gpuType, int useHyperviscosity, bool weightsComputed=false) 
            : TimeDependentPDE_CL(grid, der, comm, gpuType, weightsComputed)
        { 
            this->initialize("vortex_rollup_solve.cl"); 
        }

        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t) { 
            //DO NOTHING;
        }

    protected: 
        virtual std::string className() {return "vortex_rollup_cl";}
}; 
#endif 


