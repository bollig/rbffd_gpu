#ifndef __VORTEX_ROLLUP_H__
#define __VORTEX_ROLLUP_H__

#include "pdes/time_dependent_pde.h"


class VortexRollup : public TimeDependentPDE
{
    protected: 
        // T/F : are the weights already computed so we can avoid that cost?
        bool weightsPrecomputed;

        int useHyperviscosity; 

    public: 
        VortexRollup(Domain* grid, RBFFD* der, Communicator* comm, int useHyperviscosity, bool weightsComputed=false) 
            : TimeDependentPDE(grid, der, comm), weightsPrecomputed(weightsComputed), 
               useHyperviscosity(useHyperviscosity)
        { ; }

        // This should assemble a matrix L of weights which can be used to solve the PDE
        virtual void assemble(); 
        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t);

        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t) { 
            //DO NOTHING;
        }

    private: 
        void setupTimers(); 

    protected: 
        virtual std::string className() {return "vortex_rollup";}
}; 
#endif 

