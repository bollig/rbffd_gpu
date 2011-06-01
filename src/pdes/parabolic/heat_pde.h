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
        // T/F: do we assume uniform diffusion coefficient K in the PDE: du/dt - K Laplacian(u) = 0 
        // NOTE: if false, then we have:  du/dt - div(K .dot. grad(u)) = 0
        //       which we rewrite then as:  du/dt - ( grad(K)*grad(U) + K * laplacian(u) ) = 0
        // FIXME: the above equations assume SCALAR field. In a general vector field, the laplacian becomes: 
        //          vec_laplacian(u) = grad(div(u)) - curl(curl(u))
        //       but for now we assume Cartesian coordinates which reduces the vec_laplacian 
        //       to the same as scalar across the three variables
        bool uniformDiffusion; 

        // T/F:  laplacian(u) = du^2/dx^2 + du^2/dy^2 + du^2/dz^2. Should we
        // apply 1 set of weights for laplacian(u) directly or apply 3 sets of
        // weights to the the second derivatives independently?
        bool splitLaplacian; 

        ExactSolution* exact_ptr;

    public: 
        HeatPDE(Domain* grid, RBFFD* der, Communicator* comm, bool uniformDiffusion, bool weightsComputed=false) 
            : TimeDependentPDE(grid, der, comm), weightsPrecomputed(weightsComputed)
        { ; }

        // This should fill the solution vector with our initial conditions. 
        virtual void fillInitialConditions(ExactSolution* exact=NULL);
        virtual void fillBoundaryConditions(ExactSolution* exact=NULL);

        virtual void fillDiffusion(std::vector<SolutionType>& diff, double t);

        // This should assemble a matrix L of weights which can be used to solve the PDE
        virtual void assemble(); 
        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, double t);

        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t);

        void setSplitLaplacian(bool split) { splitLaplacian = split; }
        void setUseUniformDiffusion(bool isUniform) { uniformDiffusion = isUniform; }

    protected: 
        virtual void setupTimers(); 

        virtual std::string className() {return "heat";}
}; 
#endif 

