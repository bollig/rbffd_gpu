#ifndef __HEAT_PDE_H__
#define __HEAT_PDE_H__

#include "pdes/time_dependent_pde.h"

// TODO: extend this class and compute diffusion in two terms: lapl(y(t)) = div(y(t)) .dot. grad(y(t))
class HeatPDE : public TimeDependentPDE
{
    protected: 
        std::vector<SolutionType> boundary_values; 
        // T/F : are the weights already computed so we can avoid that cost?
        bool weightsPrecomputed;

        
        // T/F:  laplacian(u) = du^2/dx^2 + du^2/dy^2 + du^2/dz^2. Should we
        // apply 1 set of weights for laplacian(u) directly or apply 3 sets of
        // weights to the the second derivatives independently?
        bool splitLaplacian; 

        // T/F: do we assume uniform diffusion coefficient K in the PDE: du/dt - K Laplacian(u) = 0 
        // NOTE: if false, then we have:  du/dt - div(K .dot. grad(u)) = 0
        //       which we rewrite then as:  du/dt - ( grad(K)*grad(U) + K * laplacian(u) ) = 0
        // FIXME: the above equations assume SCALAR field. In a general vector field, the laplacian becomes: 
        //          vec_laplacian(u) = grad(div(u)) - curl(curl(u))
        //       but for now we assume Cartesian coordinates which reduces the vec_laplacian 
        //       to the same as scalar across the three variables
        bool uniformDiffusion; 

    public: 
        HeatPDE(Domain* grid, RBFFD* der, Communicator* comm, bool useUniformDiffusion, bool weightsComputed=false) 
            : TimeDependentPDE(grid, der, comm), weightsPrecomputed(weightsComputed),
              splitLaplacian(false), uniformDiffusion(useUniformDiffusion)
        { ; }

        // This should fill the solution vector with our initial conditions. 
        virtual void fillInitialConditions(ExactSolution* exact=NULL);
        virtual void fillBoundaryConditions(ExactSolution* exact=NULL);

        virtual void fillDiffusion(std::vector<SolutionType>& diff, std::vector<SolutionType>& sol, double t, unsigned int n_nodes);

        // This should assemble a matrix L of weights which can be used to solve the PDE
        virtual void assemble(); 
        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t);

        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t);

        void setSplitLaplacian(bool split) { splitLaplacian = split; }
        void setUseUniformDiffusion(bool isUniform) { uniformDiffusion = isUniform; }

    private: 
        void setupTimers(); 

    protected: 
        void solveDivGrad(std::vector<SolutionType>& u_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t);
        void solveRewrittenLaplacian(std::vector<SolutionType>& u_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t);
        
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes)
        { std::cout << "[HeatPDE] ERROR! SHOULD CALL THE TIME BASED SOLVE ROUTINE\n"; exit(EXIT_FAILURE); } 

        virtual void solve()
        { std::cout << "[HeatPDE] ERROR! SHOULD CALL THE TIME BASED SOLVE ROUTINE\n"; exit(EXIT_FAILURE); } 

        virtual std::string className() {return "heat";}
}; 
#endif 

