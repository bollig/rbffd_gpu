#ifndef __TIME_DEPENDENT_PDE_CL_H__
#define __TIME_DEPENDENT_PDE_CL_H__

#include "pdes/time_dependent_pde.h"

class TimeDependentPDE_CL : public TimeDependentPDE 
{
    protected: 
    // Can be overridden so subclasses compute this on the GPU
    virtual void computeUpdate() { std::cout << "TODO: TimeDependentPDE_CL::computeUpdate\n"; }

   // Can be overridden so subclasses compute on the GPU
    virtual void applyUpdate() { std::cout << "TODO: TimeDependentPDE_CL::applyUpdate\n"; } 

    virtual void advanceEuler() { std::cout << "TODO: TimeDependentPDE_CL::advanceEuler\n"; }
};
#endif // __TIME_DEPENDENT_PDE_H__
