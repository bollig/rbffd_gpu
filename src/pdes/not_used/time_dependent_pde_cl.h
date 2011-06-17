#ifndef __TIME_DEPENDENT_PDE_CL_H__
#define __TIME_DEPENDENT_PDE_CL_H__

#include "pdes/time_dependent_pde.h"
#include "utils/opencl/cl_base_class.h"

class TimeDependentPDE_CL : public TimeDependentPDE, public CLBaseClass
{
    cl::Buffer gpu_U_G;
    cl::Buffer computed_deriv_weights;

    public: 
        TimeDependentPDE_CL(Domain* grid, RBFFD* der, Communicator* comm) 
            : TimeDependentPDE(grid, der, comm), CLBaseClass(comm->getRank())
        {
#if 0
            this->setupTimers(); 
            this->loadKernel(); 
            this->allocateGPUMem();
#endif 
        }

    protected: 
        virtual void advanceFirstOrderEuler(double dt);
        virtual void advanceSecondOrderMidpoint(double dt);
        virtual void advanceRungeKutta4(double dt);
};
#endif // __TIME_DEPENDENT_PDE_H__
