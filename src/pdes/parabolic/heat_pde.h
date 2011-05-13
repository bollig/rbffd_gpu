#ifndef __HEAT_PDE_H__
#define __HEAT_PDE_H__

#include "pdes/time_dependent_pde.h"

class HeatPDE : public TimeDependentPDE
{
    public: 
        HeatPDE(Domain* grid, RBFFD* der, Communicator* comm) 
            : TimeDependentPDE(grid, der, comm) 
        { ; }

        // This should fill the solution vector with our initial conditions. 
        virtual void fillInitialConditions();

        // This should assemble a matrix L of weights which can be used to solve the PDE
        virtual void assemble(); 
        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve();

    protected: 
        virtual void setupTimers(); 

        virtual std::string className() {return "heat";}
}; 
#endif 

