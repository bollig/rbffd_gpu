#ifndef __TIME_DEPENDENT_PDE_H__
#define __TIME_DEPENDENT_PDE_H__

#include "pdes/pde.h"


// Interface class
class TimeDependentPDE : public PDE 
{
    // This count should match the number of TimeScheme types
#define NUM_TIME_SCHEMES 3
    enum TimeScheme {FIRST_EULER=0, SECOND_EULER, RK45};

    public: 
    TimeDependentPDE(Domain* grid, RBFFD* der, Communicator* comm) 
        : PDE(grid, der, comm) 
    { 
        fillInitialConditions();
    }

    // Fill in the initial conditions of the PDE. (overwrite the solution)
    virtual void fillInitialConditions();

    // Advancing requires: 
    //  - computing an update to the current solution (i.e., calling
    //  applyWeightsForDerivs(currentSolution)) 
    //  - applying the updates to the current solution (i.e., RK45 weighted
    //  summation of intermediate updates).
    //  NOTE: at the end of the advance routine the PDE::solution should
    //  contain the advanced solution. If intermediate steps are required (i.e.
    //  in 2nd order or RK45, intermediate solutions and ghost node broadcasts
    //  are required), then archive the original solution and any subsequent
    //  buffers and overwrite the final solution at the end of the routine.
    virtual void advance(TimeScheme& which);

    protected: 
    virtual void advanceFirstEuler();
    virtual void advanceSecondEuler();


};
#endif // __TIME_DEPENDENT_PDE_H__
