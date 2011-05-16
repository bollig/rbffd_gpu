#ifndef __HEAT_PDE_H__
#define __HEAT_PDE_H__

#include "pdes/time_dependent_pde.h"

// TODO: extend this class and compute diffusion in two terms: lapl(y(t)) = div(y(t)) .dot. grad(y(t))
class HeatPDE : public TimeDependentPDE
{
    private: 
        std::vector<SolutionType> boundary_values; 
        // T/F : are the weights already computed so we can avoid that cost?
        bool weightsPrecomputed;


        ExactSolution* exact_ptr;

    public: 
        HeatPDE(Domain* grid, RBFFD* der, Communicator* comm, bool weightsComputed=false) 
            : TimeDependentPDE(grid, der, comm), weightsPrecomputed(weightsComputed)
        { ; }

        // This should fill the solution vector with our initial conditions. 
        virtual void fillInitialConditions(ExactSolution* exact=NULL);
        virtual void fillBoundaryConditions(ExactSolution* exact=NULL);

        // This should assemble a matrix L of weights which can be used to solve the PDE
        virtual void assemble(); 
        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, double t);

        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t);
    protected: 
        virtual void setupTimers(); 

        virtual std::string className() {return "heat";}
}; 
#endif 

